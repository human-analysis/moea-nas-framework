# run neural architecture transfer
import os
import yaml
import pathlib
import argparse
import numpy as np

import torch

from ofa.utils import MyRandomResizedCrop

from search.algorithms.msunas import MSuNAS
from train.trainer import SuperNetTrainer
from supernets.ofa_mbnv3 import GenOFAMobileNetV3
from search.evaluators.ofa_evaluator import OFAEvaluator
from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace


parser = argparse.ArgumentParser(description='Warm-up Supernet Training')
parser.add_argument('--dataset', default='cifar10', type=str, metavar='DATASET', help='Name of dataset to train (default: "cifar10")')
parser.add_argument('--data', default='/home/vishnu/datastore/processed/', type=str, metavar='DATA', help='Path to the dataset images')
parser.add_argument('--objs', default='acc&flops', type=str, metavar='OBJ', help='which objectives to optimize, separated by "&"')
parser.add_argument('--valid-size', type=int, default=None, metavar='VS', help='number of images separated from training set to guild NAS (default: 5000)')
parser.add_argument('--save', default='.tmp', type=str, metavar='SAVE', help='path to dir for saving results')
parser.add_argument('--resume', default=None, type=str, metavar='RESUME', help='path to dir for resume of search')
args = parser.parse_args()

# dataset related settings
args.train_batch_size = 96  # input batch size for training
args.test_batch_size = 100  # input batch size for testing
args.image_sizes = [192, 224, 256]
# args.valid_size = None  # make sure to remove in production
args.workers = 4

# search related settings
args.surrogate = 'lgb'  # which surrogate model to fit accuracy predictor
args.n_doe = 100
args.n_gen = 8
args.max_gens = 30
args.topN = 150

# supernet adaption related settings
args.ft_epochs_gen = 5  # number of epochs to adapt supernet in each gen
args.epochs = int(args.ft_epochs_gen * args.max_gens)  # number of epochs in total to adapt supernet
args.lr = 2.5e-3  # initial learning rate
args.lr_min = 0.0  # final learning rate
args.momentum = 0.9  # optimizer momentum
args.wd = 3e-4  # optimizer weight decay
args.grad_clip = 5  # clip gradient norm
args.model_ema = False
args.dynamic_batch_size = 4
args.kd_ratio = 1.0
args.sub_train_size = 960
args.sub_train_batch_size = 96
args.report_freq = 10

args.save = os.path.join(args.save, args.dataset)
args.supernet_weights = os.path.join(
        pathlib.Path(__file__).parent.resolve(), args.save, 'supernet', 'checkpoint.pth.tar')
args.teacher_weights = os.path.join(
        pathlib.Path(__file__).parent.resolve(), args.save, 'supernet', 'ofa_mbv3_d4_e6_k7_w1.2', 'checkpoint.pth.tar')


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

    # construct the supernet
    supernet = GenOFAMobileNetV3(
        n_classes=data_provider.n_classes, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # load pretrained supernet checkpoints weights
    supernet_state_dicts = torch.load(args.supernet_weights, map_location='cpu')
    state_dicts = [supernet_state_dicts['model_w1.0_state_dict'],
                   supernet_state_dicts['model_w1.2_state_dict']]
    supernet.load_state_dict(state_dicts)

    # push supernet to cuda
    supernet.cuda()

    # get the teacher model
    teacher_str = search_space.decode([np.array(search_space.ub)])[0]
    supernet.set_active_subnet(**teacher_str)
    teacher = supernet.get_active_subnet(preserve_weight=False)
    teacher_state_dict = torch.load(args.teacher_weights, map_location='cpu')['model_state_dict']
    teacher.load_state_dict(teacher_state_dict)
    teacher = teacher.cuda()

    # define the supernet trainer
    trainer = SuperNetTrainer(supernet, teacher, search_space, args.epochs, args.lr, data_provider, cur_epoch=0,
                              lr_end=args.lr_min, momentum=args.momentum, wd=args.wd, save=args.save,
                              dynamic_batch_size=args.dynamic_batch_size, kd_ratio=args.kd_ratio,
                              sub_train_size=args.sub_train_size, sub_train_batch_size=args.sub_train_batch_size,
                              report_freq=args.report_freq)

    # construct the evaluator
    evaluator = OFAEvaluator(supernet, data_provider, sub_train_size=args.sub_train_size,
                             sub_train_batch_size=args.sub_train_batch_size)

    # construct MSuNAS search engine
    nas_method = MSuNAS(search_space, evaluator, trainer, objs=args.objs, surrogate=args.surrogate,
                     n_doe=args.n_doe, n_gen=args.n_gen, max_gens=args.max_gens, topN=args.topN,
                     ft_epochs_gen=args.ft_epochs_gen, save_path=args.save, resume=args.resume)

    # kick-off the search
    nas_method.search()

    with open(os.path.join(args.save, 'args.yaml'), 'w') as f:
        f.write(args_text)

    return


if __name__ == '__main__':
    main()
