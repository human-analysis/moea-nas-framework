#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import json
import random
import numpy as np
from PIL import Image
import os.path as osp

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .transform import ColorJitter, HorizontalFlip, RandomScale


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(im=im.crop(crop), lb=lb.crop(crop))


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', randomscale=(
            0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        # print('self.mode', self.mode)
        self.ignore_lb = 255

        with open('./data_providers/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        im_names = os.listdir(impth)
        names = [el.replace('_leftImg8bit.png', '') for el in im_names]
        impths = [osp.join(impth, el) for el in im_names]
        imgnames.extend(names)
        self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        lbnames = os.listdir(gtpth)
        lbnames = [el for el in lbnames if 'labelIds' in el]
        names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
        lbpths = [osp.join(gtpth, el) for el in lbnames]
        gtnames.extend(names)
        self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        # print('self.len', self.mode, self.len)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

