#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, './')

import torch

from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

from evaluation.evaluate import validate
from supernet.ofa_mbnv3 import GenOFAMobileNetV3
from search.search_spaces.ofa_search_space import MobileNetV3SearchSpace
from data_providers.imagenet import ImagenetDataProvider


def main():

    # ----------------- dataset loading ------------------- #
    batchsize = 200
    n_workers = 12
    imagenet_dataprovider = ImagenetDataProvider(
        save_path='/home/zhichao/datasets/ILSVRC2012', train_batch_size=256, test_batch_size=batchsize,
        valid_size=None, n_worker=n_workers)

    # ---------------- construct the search space and supernet ------------ #
    search_space = MobileNetV3SearchSpace()

    ofa_network = GenOFAMobileNetV3(
        n_classes=1000, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # load checkpoints weights
    state_dicts = [
        torch.load('/home/zhichao/2021/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.0',
                   map_location='cpu')['state_dict'],
        torch.load('/home/zhichao/2021/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.2',
                   map_location='cpu')['state_dict']]

    ofa_network.load_state_dict(state_dicts)

    # randomly sample a subnet
    subnet_settings = ofa_network.sample_active_subnet()
    print("sampled subnet: ")
    print(subnet_settings)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)

    # set the image scale
    imagenet_dataprovider.assign_active_img_size(subnet_settings['r'])

    dl = imagenet_dataprovider.valid
    sdl = imagenet_dataprovider.build_sub_train_loader(2000, 200)

    # reset BN running statistics of the sampled subnet
    subnet.cuda()
    set_running_statistics(subnet, sdl)

    criterion = torch.nn.CrossEntropyLoss()
    stats = validate(subnet, dl, criterion)

    print(stats)


if __name__ == "__main__":
    main()