#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import os
import time
import random
import logging
import datetime
import argparse
import warnings
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from evaluation.evaluate import MscEval
from models.fanet import FANet
from supernet.ofa_fanetplus import OFAFANetPlus
from data_providers.cityscapes import CityScapes
from train.utils import setup_logger, OhemCELoss, Optimizer

from ofa.utils import subset_mean
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

warnings.filterwarnings("ignore")
logger = logging.getLogger()

SUB_SEED = 937162211  # random seed for sampling subset


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def _print_expand_ratio(e):
    # assuming basic block OFANet
    return [np.round(np.mean(e[:3]), 1), np.round(np.mean(e[3:7]), 1),
            np.round(np.mean(e[7:13]), 1), np.round(np.mean(e[13:16]), 2)]


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--nccl_ip', type=int, default=12)
    parse.add_argument('--data', type=str, default='/home/cseadmin/huangsh/datasets/Cityscapes')
    parse.add_argument('--n_img_per_gpu', dest='n_img_per_gpu', type=int, default=16, )
    # parse.add_argument('--max_iter', dest='max_iter', type=int, default=60000, )
    parse.add_argument('--use_conv_last', dest='use_conv_last', type=str2bool, default=False,)
    parse.add_argument('--respath', dest='respath', type=str, default='../Exps', )
    parse.add_argument('--mode', dest='mode', type=str, default='train', )
    parse.add_argument('--ckpt', dest='ckpt', type=str, default=None,
                       help="path to the checkpoint weights")
    parse.add_argument('--backbone', dest='backbone', type=str, default='basic',
                       help="which backbone architecture to construct FANet")
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    parse.add_argument('--n_workers_train', dest='n_workers_train', type=int, default=12, )
    parse.add_argument('--n_workers_val', dest='n_workers_val', type=int, default=2, )
    # progressive training related settings
    parse.add_argument('--task', dest='task', type=str, default='depth', choices=['depth', 'expand', 'width'],
                       help="which task to train ofa-fanet")
    parse.add_argument('--phase', dest='phase', type=int, default=1,
                       help="which phase within current task")
    return parse.parse_args()


def evaluate(ofa_net, dl, sdl, scale, subnets):

    ofa_net.eval()
    mIoUs = []
    for subnet in subnets:
        subnet_str = ','.join(['%s_%s' % (
            key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
        ) for key, val in subnet.items()])
        ofa_net.module.cp.backbone.set_active_subnet(d=subnet['depth'], e=subnet['expand'], w=subnet['width'])
        # reset BN running statistics
        set_running_statistics(ofa_net, sdl)
        with torch.no_grad():
            single_scale = MscEval(scale=scale)
            mIoU = single_scale(ofa_net, dl, 19)
        if dist.get_rank() == 0:
            print("{} mIoU = {:.4f}".format(subnet_str, mIoU))
        mIoUs.append(mIoU)

    return np.mean(mIoUs)


def train_one_iter(
        it, im, lb, dynamic_batch_size, ofa_net, teacher,
        optim, n_min, criteria_p, criteria_16, criteria_32, criterion_kd, task='depth', lamb=0.2):

    # switch to train mode
    ofa_net.train()

    im = im.cuda()
    lb = lb.cuda()
    H, W = im.size()[2:]
    lb = torch.squeeze(lb, 1)

    # soft outputs from teacher
    with torch.no_grad():
        soft_out, _, _, _ = teacher(im)

    # clean gradients
    optim.zero_grad()

    loss_of_subnets = 0
    # compute output
    subnet_str = ''
    for _ in range(dynamic_batch_size):
        # set random seed before sampling
        subnet_seed = int('%d%.3d%.3d' % (it, _, 0))
        random.seed(subnet_seed)
        subnet_settings = ofa_net.module.sample_active_subnet()
        if task == 'depth':
            subnet_str += '({:d})-> '.format(_) + 'd: ' \
                          + '_'.join(map(str, subnet_settings['d'])) \
                          + ', e: {:.1f}'.format(np.mean(subnet_settings['e'])) \
                          + ', w: {:.1f};  '.format(np.mean(subnet_settings['w']))
        elif task == 'expand':
            subnet_str += '({:d})-> '.format(_) + 'd: ' \
                          + '_'.join(map(str, subnet_settings['d'])) \
                          + ', e: {:.4f}'.format(map(str, _print_expand_ratio(subnet_settings['e']))) \
                          + ', w: {:.1f};  '.format(np.mean(subnet_settings['w']))
        else:
            subnet_str += '({:d})-> '.format(_) + 'd: ' \
                          + '_'.join(map(str, subnet_settings['d'])) \
                          + ', e: ' + '_'.join(map(str, _print_expand_ratio(subnet_settings['e']))) \
                          + ', w: ' + '_'.join(map(str, subnet_settings['w'])) + ';  '

        out, out16, out32 = ofa_net(im)
        lossp = criteria_p(out, lb, n_min)
        loss2 = criteria_16(out16, lb, n_min)
        loss3 = criteria_32(out32, lb, n_min)
        loss = lossp + lamb * loss2 + lamb * loss3  # taken from
        # https://github.com/VITA-Group/FasterSeg/blob/478b0265eb9ab626cfbe503ad16d2452878b38cc/train/train.py#L256
        loss = loss + criterion_kd(F.softmax(out, dim=1).log(), F.softmax(soft_out, dim=1))

        # measure accuracy and record loss
        loss_of_subnets += loss.item()

        loss.backward()

    # if dist.get_rank() == 0:
    #     print(subnet_str)

    # backprob the accumulated gradients
    optim.step()

    return loss_of_subnets / dynamic_batch_size


def main():
    args = parse_args()

    exp_dir = 'ofa_plus_{}_'.format(args.backbone)

    if args.task == 'depth':
        exp_dir += 'depth@phase{}'.format(args.phase)
        dynamic_batch_size = 2

        expand_constr, width_constr = [1.0], [2]
        if args.phase == 1:
            depth_constr = [1, 1, 2, 1]
            lr_start = 2.5e-3
            warmup_steps = 5
            max_iter = 12000
            save_iter_start = 6000
            save_iter_sep = 200
            # resuming from a checkpoint stopped at 6400 iterations
            # lr_start = 0.000588
            # warmup_steps = -1
            # max_iter = 2600
            # save_iter_start = 0

        else:
            depth_constr = [0, 0, 0, 0]
            lr_start = 1e-2  # 7.5e-3
            warmup_steps = 500
            max_iter = 60000
            save_iter_start = 30000
            save_iter_sep = 1000

        subnets2eval = [
            {'depth': depth_constr, 'expand': [1.0] * 16, 'width': [2] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [1.0] * 16, 'width': [2] * 5}]

    elif args.task == 'expand':
        exp_dir += 'depth-expand@phase{}'.format(args.phase)
        dynamic_batch_size = 4

        depth_constr, width_constr = [0, 0, 0, 0], [2]
        if args.phase == 1:
            expand_constr = [0.8, 1.0]
            lr_start = 2.5e-3
            warmup_steps = 5
            max_iter = 12000
            save_iter_start = 6000
            save_iter_sep = 200
        else:
            expand_constr = [0.65, 0.8, 1.0]
            lr_start = 1e-2  # 7.5e-3
            warmup_steps = 500
            max_iter = 60000
            save_iter_start = 30000
            save_iter_sep = 1000

        subnets2eval = [
            {'depth': [0, 0, 0, 0], 'expand': [expand_constr[0]] * 16, 'width': [2] * 5},
            {'depth': [0, 0, 0, 0], 'expand': [expand_constr[-1]] * 16, 'width': [2] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [expand_constr[0]] * 16, 'width': [2] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [expand_constr[-1]] * 16, 'width': [2] * 5}]

    elif args.task == 'width':
        exp_dir += 'depth-expand-width@phase{}'.format(args.phase)
        dynamic_batch_size = 4

        depth_constr, expand_constr = [0, 0, 0, 0], [0.65, 0.8, 1.0]
        if args.phase == 1:
            width_constr = [1, 2]
            lr_start = 2.5e-3
            warmup_steps = 5
            max_iter = 12000
            save_iter_start = 6000
            save_iter_sep = 200

        else:
            width_constr = [0, 1, 2]
            lr_start = 1e-2  # 7.5e-3
            warmup_steps = 500
            max_iter = 80000
            save_iter_start = 40000
            save_iter_sep = 1000

        subnets2eval = [
            {'depth': [0, 0, 0, 0], 'expand': [0.65] * 16, 'width': [width_constr[0]] * 5},
            {'depth': [0, 0, 0, 0], 'expand': [0.65] * 16, 'width': [width_constr[-1]] * 5},
            {'depth': [0, 0, 0, 0], 'expand': [1.0] * 16, 'width': [width_constr[0]] * 5},
            {'depth': [0, 0, 0, 0], 'expand': [1.0] * 16, 'width': [width_constr[-1]] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [0.65] * 16, 'width': [width_constr[0]] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [0.65] * 16, 'width': [width_constr[-1]] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [1.0] * 16, 'width': [width_constr[0]] * 5},
            {'depth': [2, 3, 4, 2], 'expand': [1.0] * 16, 'width': [width_constr[-1]] * 5}]

    else:
        raise NotImplementedError

    save_dir = osp.join(args.respath, exp_dir)

    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:332{}'.format(args.nccl_ip),
                            world_size=torch.cuda.device_count(), rank=args.local_rank)

    setup_logger(save_dir)
    # dataset
    args.n_classes = 19
    args.output_aux = True
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val

    mode = args.mode

    cropsize = [1024, 512]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    if dist.get_rank() == 0:
        logger.info('n_workers_train: {}'.format(n_workers_train))
        logger.info('n_workers_val: {}'.format(n_workers_val))
        logger.info('mode: {}'.format(args.mode))

    ds = CityScapes(args.data, cropsize=cropsize, mode=mode, randomscale=randomscale)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=n_img_per_gpu, shuffle=False, sampler=sampler,
                    num_workers=n_workers_train, pin_memory=False, drop_last=True)

    dsval = CityScapes(args.data, mode='val', randomscale=randomscale)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dlval = DataLoader(dsval, batch_size=2, shuffle=False, sampler=sampler_val,
                       num_workers=n_workers_val, drop_last=False)

    # build a subset train loader for resetting BN running statistics
    g = torch.Generator()
    g.manual_seed(SUB_SEED)
    rand_indexes = torch.randperm(ds.len, generator=g).tolist()[:500]
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_indexes)
    sdl = DataLoader(ds, batch_size=n_img_per_gpu, shuffle=False, sampler=sub_sampler,
                     num_workers=n_workers_train, pin_memory=False, drop_last=True)

    # model
    ignore_idx = 255

    # -------------------------------------------------- #
    ofa_network = OFAFANetPlus(
        backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
        width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
        output_aux=True)

    # load checkpoints weights
    ofa_network.load_state_dict(torch.load(args.ckpt, map_location='cpu'))

    # configure ofa-fanet according to different task and phase
    ofa_network.cp.backbone.set_constraint(depth_constr, constraint_type='depth')
    ofa_network.cp.backbone.set_constraint(expand_constr, constraint_type='expand_ratio')
    ofa_network.cp.backbone.set_constraint(width_constr, constraint_type='width_mult')
    # -------------------------------------------------- #

    ofa_network.cuda()
    ofa_network = nn.parallel.DistributedDataParallel(
        ofa_network, device_ids=[args.local_rank, ], output_device=args.local_rank, find_unused_parameters=True)

    # evaluate the ofa_network before training
    init_mIoUs = evaluate(ofa_network, dlval, sdl, scale=0.5, subnets=subnets2eval)

    # setup the teacher network
    teacher = FANet(pretrained_backbone=False, output_aux=True)
    teacher.load_state_dict(torch.load("./pretrained/fanet/teacher_r101d.pth", map_location='cpu'))
    teacher.cuda()
    teacher.eval()
    teacher = nn.parallel.DistributedDataParallel(
        teacher, device_ids=[args.local_rank, ], output_device=args.local_rank, find_unused_parameters=True)

    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    criterion_kd = nn.KLDivLoss()

    # optimizer
    maxmIOU50 = init_mIoUs
    momentum = 0.9
    weight_decay = 5e-4
    # lr_start = 1e-2
    # max_iter = args.max_iter
    # save_iter_sep = args.save_iter_sep
    power = 0.9
    # warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5

    if dist.get_rank() == 0:
        print('max_iter: ', max_iter)
        print('save_iter_sep: ', save_iter_sep)
        print('warmup_steps: ', warmup_steps)

    optim = Optimizer(model=ofa_network.module, loss=None, lr0=lr_start, momentum=momentum, wd=weight_decay,
                      warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, max_iter=max_iter, power=power)

    # train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()

    diter = iter(dl)

    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)

        # apply training for one iteration
        loss_of_subnets = train_one_iter(it, im, lb, dynamic_batch_size, ofa_network, teacher, optim,
                                         n_min, criteria_p, criteria_16, criteria_32, criterion_kd)

        loss_avg.append(loss_of_subnets)

        # print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            msg = ', '.join(
                ['it: {it}/{max_it}', 'lr: {lr:4f}', 'loss: {loss:.4f}', 'eta: {eta}', 'time: {time:.4f}',
                 ]).format(it=it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta)

            logger.info(msg)
            loss_avg = []
            st = ed

        if ((it + 1) >= save_iter_start) and ((it + 1) % save_iter_sep == 0):
            logger.info('evaluating the model ...')
            # logger.info('setup and restore model')
            mIOU50 = evaluate(ofa_network, dlval, sdl, scale=0.5, subnets=subnets2eval)

            # save_pth = osp.join(save_dir, 'model_iter{}_mIOU50_{}.pth'.format(
            #     it + 1, str(round(mIOU50, 4))))
            #
            # state = ofa_network.module.state_dict() if hasattr(ofa_network, 'module') else ofa_network.state_dict()
            # if dist.get_rank() == 0:
            #     torch.save(state, save_pth)
            #
            # logger.info('training iteration {}, model saved to: {}'.format(it + 1, save_pth))

            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
                save_pth = osp.join(save_dir, 'model_maxmIOU50.pth'.format(it + 1))
                state = ofa_network.module.state_dict() if hasattr(ofa_network, 'module') else ofa_network.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state, save_pth)

                logger.info('max mIOU model saved to: {}'.format(save_pth))

            logger.info('current mean mIOU50 is: {}'.format(mIOU50))
            logger.info('best mean mIOU50 is: {}.'.format(maxmIOU50))

    # dump the final model
    save_pth = osp.join(save_dir, 'model_final.pth')
    ofa_network.cpu()
    state = ofa_network.module.state_dict() if hasattr(ofa_network, 'module') else ofa_network.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)


if __name__ == "__main__":
    main()
