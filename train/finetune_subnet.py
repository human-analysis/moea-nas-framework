#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import os
import time
import json
import logging
import datetime
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.fanet import FANet
from models.fanetplus import FANetPlus
from supernet.ofa_fanet import OFAFANet
from supernet.ofa_fanetplus import OFAFANetPlus
from data_providers.cityscapes import CityScapes
from train.utils import setup_logger, OhemCELoss, MscEvalV0, Optimizer


logger = logging.getLogger()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--nccl_ip', type=int, default=12)
    parse.add_argument('--data', type=str, default='/home/cseadmin/huangsh/datasets/Cityscapes')
    parse.add_argument('--n_img_per_gpu', dest='n_img_per_gpu', type=int, default=16, )
    parse.add_argument('--max_iter', dest='max_iter', type=int, default=2000, )
    parse.add_argument('--use_conv_last', dest='use_conv_last', type=str2bool, default=False,)
    parse.add_argument('--respath', dest='respath', type=str, default='../Exps', )
    parse.add_argument('--save_iter_sep', dest='save_iter_sep', type=int, default=100, )
    parse.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=-1, )
    parse.add_argument('--mode', dest='mode', type=str, default='train', )
    parse.add_argument('--backbone', dest='backbone', type=str, default='basic',
                       choices=['basic', 'inverted', 'bottleneck'])
    parse.add_argument('--ckpt', type=str, default=None, help="path to the pretrained weights")
    parse.add_argument('--subnet', type=str, default=None, help="path to the subnet config json file")
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    parse.add_argument('--n_workers_train', dest='n_workers_train', type=int, default=12, )
    parse.add_argument('--n_workers_val', dest='n_workers_val', type=int, default=1, )
    parse.add_argument('--kd', action='store_true', default=False)
    parse.add_argument('--scale', type=float, default=0.5,
                       help='resolution scale 0.5 -> 512x1024 for Cityscapes')
    return parse.parse_args()


def train():
    args = parse_args()
    save_pth_path = osp.join(args.respath, 'ofa_fanetplus_{}'.format(args.backbone))

    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path, exist_ok=True)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:332{}'.format(args.nccl_ip),
                            world_size=torch.cuda.device_count(), rank=args.local_rank)

    setup_logger(save_pth_path)
    # dataset
    args.n_classes = 19
    args.output_aux = True
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val

    mode = args.mode

    if args.scale == 0.5:
        cropsize = [1024, 512]
        randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    elif args.scale == 0.625:
        cropsize = [1280, 640]
        randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    else:
        raise NotImplementedError

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

    # model
    ignore_idx = 255

    if args.backbone == 'basic':

        supernet = OFAFANetPlus(
            'basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0], width_mult_list=[0.65, 0.8, 1.0],
            feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]], output_aux=True)

    elif args.backbone == 'bottleneck':
        supernet = OFAFANetPlus(
            backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
            width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]],
            output_aux=True)

    elif args.backbone == 'inverted':
        supernet = OFAFANetPlus(
            'inverted', depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6],
            width_mult_list=1.0, feature_dim_list=[[40], [96], [320]], output_aux=True)

    else:
        raise NotImplementedError

    supernet.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    subnet_config = json.load(open(args.subnet, 'r'))[0]['config']
    supernet.set_active_subnet(**subnet_config)
    net = supernet.get_active_subnet(preserve_weight=True, output_aux=True)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank, ], output_device=args.local_rank, find_unused_parameters=True)

    if args.kd:
        teacher = FANet(pretrained_backbone=False, output_aux=True)
        state_dict = torch.load("./pretrained/fanet/teacher_r101d.pth", map_location='cpu')
        teacher.load_state_dict(state_dict)
        teacher.cuda()
        teacher.eval()  # ?
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.local_rank, ], output_device=args.local_rank, find_unused_parameters=True)
        criterion_kd = nn.KLDivLoss()

    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    # criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    criteria_p = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    #

    # optimizer
    maxmIOU = 0.
    # maxmIOU75 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    # lr_start = 1e-2
    lr_start = 1e-3
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5

    if dist.get_rank() == 0:
        print('max_iter: ', max_iter)
        print('save_iter_sep: ', save_iter_sep)
        print('warmup_steps: ', warmup_steps)

    optim = Optimizer(model=net.module, loss=None, lr0=lr_start, momentum=momentum, wd=weight_decay,
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

        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = criteria_p(out, lb, n_min)
        loss2 = criteria_16(out16, lb, n_min)
        loss3 = criteria_32(out32, lb, n_min)
        loss = lossp + 0.2*loss2 + 0.2*loss3

        if args.kd:
            with torch.no_grad():
                soft_out, _, _ = teacher(im)
            loss = loss + criterion_kd(F.softmax(out, dim=1).log(), F.softmax(soft_out, dim=1))

        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

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

        if (it + 1) % save_iter_sep == 0:
        # if it > 1:
            ## model
            logger.info('evaluating the model ...')
            logger.info('setup and restore model')
            net.eval()
            # ## evaluator
            logger.info('compute the mIOU')
            with torch.no_grad():
                single_scale1 = MscEvalV0(scale=args.scale)
                mIOU = single_scale1(net, dlval, args.n_classes)
                # single_scale2 = MscEvalV0(scale=0.75)
                # mIOU75 = single_scale2(net, dlval, args.n_classes)

            # save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}.pth'.format(
            #     it + 1, str(round(mIOU50, 4)), str(round(mIOU75, 4))))
            save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU@{}x{}_{}.pth'.format(
                it + 1, cropsize[1], cropsize[0], str(round(mIOU, 4))))

            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank() == 0:
                torch.save(state, save_pth)

            logger.info('training iteration {}, model saved to: {}'.format(it + 1, save_pth))

            if mIOU > maxmIOU:
                maxmIOU = mIOU
                save_pth = osp.join(save_pth_path, 'model_maxmIOU.pth'.format(it + 1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state, save_pth)

                logger.info('max mIOU model saved to: {}'.format(save_pth))

            # if mIOU75 > maxmIOU75:
            #     maxmIOU75 = mIOU75
            #     save_pth = osp.join(save_pth_path, 'model_maxmIOU75.pth'.format(it + 1))
            #     state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            #     if dist.get_rank() == 0:
            #         torch.save(state, save_pth)
            #     logger.info('max mIOU model saved to: {}'.format(save_pth))

            logger.info('mIOU is: {}, maxmIOU50 is: {}'.format(mIOU, maxmIOU))
            # logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))

            net.train()

    # dump the final model
    save_pth = osp.join(save_pth_path, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)


if __name__ == "__main__":
    train()
