# before running NSGANetV2 or NAT, run this script to first to ensure better performance
import os
import yaml
import pathlib
import argparse
import numpy as np

import torch

from ofa.utils import MyRandomResizedCrop

from supernets.utils import reset_classifier
from supernets.ofa_mbnv3 import GenOFAMobileNetV3
from train.trainer import Trainer, SuperNetTrainer
from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace

parser = argparse.ArgumentParser(description='Warm-up Supernet Training')
parser.add_argument('--dataset', default='cifar10', type=str, metavar='DATASET',
                    help='Name of dataset to train (default: "cifar10")')
parser.add_argument('--data', default='~/datasets', type=str, metavar='DATA',
                    help='Path to the dataset images')
parser.add_argument('--valid-size', type=int, default=None, metavar='VS',
                    help='number of images separated from training set to guild NAS (default: 5000)')
parser.add_argument('--phase', type=int, default=1, metavar='P',
                    help='which training phase to run (default: 1)')
parser.add_argument('--train-batch-size', type=int, default=96, metavar='TRBS',
                    help='Training batch size')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='TSBS',
                    help='Testing batch size')
parser.add_argument('--save', default='.tmp', type=str, metavar='SAVE',
                    help='path to dir for saving results')
args = parser.parse_args()

# dataset related settings
# args.train_batch_size = 96  # input batch size for training
# args.test_batch_size = 100  # input batch size for testing
args.image_sizes = [192, 224, 256]
args.workers = 4

# training related settings
if args.phase < 3:
    args.epochs = 100  # number of epochs to train
    args.lr = 7.5e-3  # initial learning rate
    args.lr_min = 0.0  # final learning rate
    args.lr_warmup_epochs = 5  # number of epochs to warm-up learning rate
    args.momentum = 0.9  # optimizer momentum
    args.wd = 3e-4  # optimizer weight decay
    args.grad_clip = 5  # clip gradient norm
    args.model_ema = False
    if args.phase < 2:
        args.save = os.path.join(args.save, args.dataset, 'supernet', 'ofa_mbv3_d4_e6_k7_w1.0')
    else:
        args.save = os.path.join(args.save, args.dataset, 'supernet', 'ofa_mbv3_d4_e6_k7_w1.2')

else:
    args.epochs = 150  # number of epochs to train
    args.lr = 7.5e-3  # 0.025  # initial learning rate
    args.lr_min = 0.0  # final learning rate
    args.lr_warmup_epochs = 10  # number of epochs to warm-up learning rate
    args.momentum = 0.9  # optimizer momentum
    args.wd = 3e-4  # optimizer weight decay
    args.grad_clip = 5  # clip gradient norm
    args.model_ema = False
    args.dynamic_batch_size = 4
    args.kd_ratio = 1.0
    args.sub_train_size = 10 * args.train_batch_size
    args.sub_train_batch_size = args.train_batch_size
    args.report_freq = 10
    args.save = os.path.join(args.save, args.dataset, 'supernet')
    args.teacher_weights = os.path.join(
        pathlib.Path(__file__).parent.resolve(), args.save, 'ofa_mbv3_d4_e6_k7_w1.2', 'checkpoint.pth.tar')


def main():
    # Cache the args as a text string to save them in the output dir
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    # construct the data provider
    if args.dataset == 'cifar10':
        from data_providers.cifar import CIFAR10DataProvider as DataProvider

    elif args.dataset == 'cifar100':
        from data_providers.cifar import CIFAR100DataProvider as DataProvider

    elif args.dataset == 'food':
        from data_providers.food101 import Food101DataProvider as DataProvider

    elif args.dataset == 'flowers':
        from data_providers.flowers102 import Flowers102DataProvider as DataProvider

    else:
        raise NotImplementedError

    data_provider = DataProvider(
        args.data, args.train_batch_size, args.test_batch_size, args.valid_size,
        n_worker=args.workers, resize_scale=1.0, distort_color=None, image_size=args.image_sizes,
        num_replicas=None, rank=None)

    MyRandomResizedCrop.CONTINUOUS = True
    MyRandomResizedCrop.SYNC_DISTRIBUTED = True

    # construct the search space
    search_space = OFAMobileNetV3SearchSpace()

    # todo: add support to resume training
    if args.phase < 3:
        # construct the supernet (assuming pretrained on ImageNet)
        supernet = GenOFAMobileNetV3(
            n_classes=1000, dropout_rate=0, image_scale_list=search_space.image_scale_list,
            width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
            expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

        # load ImageNet pretrained checkpoints weights
        state_dicts = [
            torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(),
                                    'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.0'),
                       map_location='cpu')['state_dict'],
            torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(),
                                    'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.2'),
                       map_location='cpu')['state_dict']]

        supernet.load_state_dict(state_dicts)
        for model in supernet.engine:
            reset_classifier(model, n_classes=data_provider.n_classes)  # change the task-specific layer accordingly

        # pretrained the full capacity supernet as teacher
        arch = search_space.decode([np.array(search_space.ub)])[0]
        if args.phase < 2:
            arch['w'] = 1.0
        print(arch)

        # set the teacher architecture
        supernet.set_active_subnet(**arch)
        teacher = supernet.get_active_subnet(preserve_weight=True)
        teacher = teacher.cuda()

        # define the trainer
        trainer = Trainer(teacher, args.epochs, args.lr, data_provider, cur_epoch=0, lr_end=args.lr_min,
                          lr_warmup_epochs=args.lr_warmup_epochs, momentum=args.momentum, wd=args.wd, save=args.save)

        # kick-off the training
        trainer.train()

    else:
        # construct the supernet (assuming pretrained on ImageNet)
        supernet = GenOFAMobileNetV3(
            n_classes=data_provider.n_classes, dropout_rate=0, image_scale_list=search_space.image_scale_list,
            width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
            expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

        # load ImageNet pretrained checkpoints weights
        state_dicts = [
            torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(), args.save,
                                    'ofa_mbv3_d4_e6_k7_w1.0', 'checkpoint.pth.tar'),
                       map_location='cpu')['model_state_dict'],
            torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(), args.save,
                                    'ofa_mbv3_d4_e6_k7_w1.2', 'checkpoint.pth.tar'),
                       map_location='cpu')['model_state_dict']]

        supernet.load_state_dict(state_dicts)
        for engine in supernet.engine:
            engine.re_organize_middle_weights(expand_ratio_stage=1)

        # warm-up subparts of supernet by uniform sampling
        # push supernet to cuda
        supernet.cuda()

        # get the teacher model
        teacher_str = search_space.decode([np.array(search_space.ub)])[0]
        supernet.set_active_subnet(**teacher_str)
        teacher = supernet.get_active_subnet(preserve_weight=False)
        teacher_state_dict = torch.load(args.teacher_weights, map_location='cpu')['model_state_dict']
        teacher.load_state_dict(teacher_state_dict)
        teacher = teacher.cuda()

        distributions = None
        # # construct the distribution, just for debug
        # import json
        # from search.algorithms.evo_nas import EvoNAS
        # from search.algorithms.utils import distribution_estimation, rank_n_crowding_survival
        #
        # archive = json.load(open(os.path.join(
        #     pathlib.Path(__file__).parent.resolve(),
        #     "tmp/MobileNetV3SearchSpaceNSGANetV2-acc&flops-lgb-n_doe@100-n_iter@8-max_iter@30/"
        #     "iter_30/archive.json"), 'r'))
        # archs = [m['arch'] for m in archive]
        # X = search_space.encode(archs)
        # F = np.array(EvoNAS.get_attributes(archive, _attr='err&flops'))
        #
        # print(X.shape)
        # print(F.shape)
        #
        # sur_X = rank_n_crowding_survival(X, F, n_survive=150)
        # distributions = []
        # for j in range(X.shape[1]):
        #     distributions.append(distribution_estimation(sur_X[:, j]))

        # define the trainer
        trainer = SuperNetTrainer(supernet, teacher, search_space, args.epochs, args.lr, data_provider, cur_epoch=0,
                                  lr_end=args.lr_min, lr_warmup_epochs=args.lr_warmup_epochs, momentum=args.momentum,
                                  wd=args.wd, save=args.save, dynamic_batch_size=args.dynamic_batch_size,
                                  kd_ratio=args.kd_ratio, sub_train_size=args.sub_train_size,
                                  sub_train_batch_size=args.sub_train_batch_size, distributions=distributions,
                                  report_freq=args.report_freq)

        # kick-off the training
        trainer.train()

    with open(os.path.join(args.save, 'args.yaml'), 'w') as f:
        f.write(args_text)

    return


if __name__ == '__main__':
    main()
