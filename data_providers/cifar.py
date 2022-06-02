import os
import numpy as np

import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from cutmix.cutmix import CutMix
from data_providers.auto_augment import AutoAugment
from ofa.imagenet_classification.data_providers.base_provider import DataProvider
from ofa.utils import MyRandomResizedCrop, MyDistributedSampler


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CIFAR10DataProvider(DataProvider):
    
    def __init__(self, save_path=None, train_batch_size=96, test_batch_size=100, valid_size=None,
                 n_worker=2, resize_scale=1.0, distort_color=None, image_size=224, num_replicas=None, rank=None):

        self._save_path = save_path
        
        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self.valid_size = valid_size

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from ofa.utils.my_dataloader import MyDataLoader
            self.image_size.sort()  # e.g., 192 -> 256
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)
        
        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.data) * valid_size)
            
            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.data), valid_size)
            
            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            
            self.train = train_loader_class(
                CutMix(train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                # train_dataset,
                batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            # todo: in case no valid size is provided, we use new cifar10 test
            valid_dataset = self.valid_dataset(valid_transforms)
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                self.train = train_loader_class(
                    CutMix(train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                    # train_dataset,
                    batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True
                )

                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas, rank)
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size, sampler=valid_sampler, num_workers=n_worker,
                    pin_memory=True,)
            else:
                self.train = train_loader_class(
                    CutMix(train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                    # train_dataset,
                    batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,)

            # self.valid = None
        
        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )
        
        # if self.valid is None:
        #     self.valid = self.test
    
    @staticmethod
    def name():
        return 'cifar10'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return 10
    
    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/mnt/datastore/CIFAR'  # home server

            if not os.path.exists(self._save_path):
                self._save_path = '/mnt/datastore/CIFAR'  # home server
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        # dataset = datasets.ImageFolder(self.train_path, _transforms)
        dataset = torchvision.datasets.CIFAR10(
            root=self.train_path, train=True, download=False, transform=_transforms)
        return dataset

    def valid_dataset(self, _transforms):
        from data_providers.datasets.cifar import AdditioanlCIFAR10
        dataset = AdditioanlCIFAR10(root=self.valid_path, transform=_transforms)
        return dataset

    def test_dataset(self, _transforms):
        # dataset = datasets.ImageFolder(self.valid_path, _transforms)
        dataset = torchvision.datasets.CIFAR10(
            root=self.test_path, train=False, download=False, transform=_transforms)
        return dataset
    
    @property
    def train_path(self):
        # return os.path.join(self.save_path, 'train')
        return self.save_path

    @property
    def valid_path(self):
        # return os.path.join(self.save_path, 'train')
        return os.path.join(self.save_path, 'Additional-CIFAR-10')

    @property
    def test_path(self):
        # return os.path.join(self.save_path, 'val')
        return self.save_path
    
    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])

    # cifar type transform
    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, valid_set_size: %s' % (self.distort_color, self.valid_size))

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
            img_size_min = min(image_size)
        else:
            resize_transform_class = transforms.RandomResizedCrop
            img_size_min = image_size

        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0), ratio=(1.0, 1.0)),
            # transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            self.normalize,
        ]

        # train_transforms += [
        #     transforms.ToTensor(),
        #     # Cutout(length=56),
        #     self.normalize,
        # ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            # transforms.Resize(int(math.ceil(image_size / 0.875))),
            # transforms.CenterCrop(image_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]
    
    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers
            
            # n_samples = len(self.train.dataset.data)
            n_samples = len(self.train.dataset.dataset.data)  # for cutmix dataset

            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            
            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)

            sub_data_loader = torch.utils.data.DataLoader(
                # new_train_dataset,
                CutMix(new_train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]


class CIFAR100DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=96, test_batch_size=256, valid_size=None,
                 n_worker=2, resize_scale=1.0, distort_color=None, image_size=224, num_replicas=None, rank=None):

        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self.valid_size = valid_size

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from ofa.utils.my_dataloader import MyDataLoader
            self.image_size.sort()  # e.g., 192 -> 256
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.data) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.data), valid_size)

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.train = train_loader_class(
                CutMix(train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                # train_dataset,
                batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            # todo: in case no valid size is provided, an error will be triggered
            # valid_dataset = self.valid_dataset(valid_transforms)
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                self.train = train_loader_class(
                    CutMix(train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                    # train_dataset,
                    batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True
                )

                # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas, rank)
                # self.valid = torch.utils.data.DataLoader(
                #     valid_dataset, batch_size=test_batch_size, sampler=valid_sampler, num_workers=n_worker,
                #     pin_memory=True, )
            else:
                self.train = train_loader_class(
                    CutMix(train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                    # train_dataset,
                    batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
                # self.valid = torch.utils.data.DataLoader(
                #     valid_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True, )

            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar100'

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/mnt/datastore/CIFAR'  # home server

            if not os.path.exists(self._save_path):
                self._save_path = '/mnt/datastore/CIFAR'  # home server
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(
            root=self.train_path, train=True, download=False, transform=_transforms)
        return dataset

    def valid_dataset(self, _transforms):
        raise NotImplementedError

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(
            root=self.test_path, train=False, download=False, transform=_transforms)
        return dataset

    @property
    def train_path(self):
        return self.save_path

    @property
    def valid_path(self):
        raise NotImplementedError

    @property
    def test_path(self):
        return self.save_path

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])

    # cifar type transform
    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, valid_set_size: %s' % (self.distort_color, self.valid_size))

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
            img_size_min = min(image_size)
        else:
            resize_transform_class = transforms.RandomResizedCrop
            img_size_min = image_size

        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0), ratio=(1.0, 1.0)),
            # transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        try:
            self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        except AttributeError:
            pass
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            # n_samples = len(self.train.dataset.data)
            n_samples = len(self.train.dataset.dataset.data)  # for cutmix dataset

            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)

            sub_data_loader = torch.utils.data.DataLoader(
                # new_train_dataset,
                CutMix(new_train_dataset, self.n_classes, beta=1.0, prob=0.5, num_mix=2),
                batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]


# class CINIC10DataProvider(DataProvider):
#
#     def __init__(self, save_path=None, train_batch_size=96, test_batch_size=256, valid_size=None,
#                  n_worker=2, resize_scale=0.08, distort_color=None, image_size=224, num_replicas=None, rank=None):
#
#         self._save_path = save_path
#
#         self.image_size = image_size  # int or list of int
#         self.distort_color = distort_color
#         self.resize_scale = resize_scale
#
#         self._valid_transform_dict = {}
#         if not isinstance(self.image_size, int):
#             assert isinstance(self.image_size, list)
#             from codebase.data_providers.my_data_loader import MyDataLoader
#             self.image_size.sort()  # e.g., 160 -> 224
#             MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
#             MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)
#
#             for img_size in self.image_size:
#                 self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
#             self.active_img_size = max(self.image_size)
#             valid_transforms = self._valid_transform_dict[self.active_img_size]
#             train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
#         else:
#             self.active_img_size = self.image_size
#             valid_transforms = self.build_valid_transform()
#             train_loader_class = torch.utils.data.DataLoader
#
#         train_transforms = self.build_train_transform()
#         train_dataset = self.train_dataset(train_transforms)
#
#         if valid_size is not None:
#             if not isinstance(valid_size, int):
#                 assert isinstance(valid_size, float) and 0 < valid_size < 1
#                 valid_size = int(len(train_dataset.data) * valid_size)
#
#             valid_dataset = self.train_dataset(valid_transforms)
#             train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.data), valid_size)
#
#             if num_replicas is not None:
#                 train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
#                 valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
#             else:
#                 train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
#                 valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
#
#             self.train = train_loader_class(
#                 train_dataset, batch_size=train_batch_size, sampler=train_sampler,
#                 num_workers=n_worker, pin_memory=True,
#             )
#             self.valid = torch.utils.data.DataLoader(
#                 valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
#                 num_workers=n_worker, pin_memory=True,
#             )
#         else:
#             if num_replicas is not None:
#                 train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
#                 self.train = train_loader_class(
#                     train_dataset, batch_size=train_batch_size, sampler=train_sampler,
#                     num_workers=n_worker, pin_memory=True
#                 )
#             else:
#                 self.train = train_loader_class(
#                     train_dataset, batch_size=train_batch_size, shuffle=True,
#                     num_workers=n_worker, pin_memory=True,
#                 )
#             self.valid = None
#
#         test_dataset = self.test_dataset(valid_transforms)
#         if num_replicas is not None:
#             test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
#             self.test = torch.utils.data.DataLoader(
#                 test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
#             )
#         else:
#             self.test = torch.utils.data.DataLoader(
#                 test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
#             )
#
#         if self.valid is None:
#             self.valid = self.test
#
#     @staticmethod
#     def name():
#         return 'cinic10'
#
#     @property
#     def data_shape(self):
#         return 3, self.active_img_size, self.active_img_size  # C, H, W
#
#     @property
#     def n_classes(self):
#         return 10
#
#     @property
#     def save_path(self):
#         if self._save_path is None:
#             self._save_path = '/mnt/datastore/CINIC10'  # home server
#
#             if not os.path.exists(self._save_path):
#                 self._save_path = '/mnt/datastore/CINIC10'  # home server
#         return self._save_path
#
#     @property
#     def data_url(self):
#         raise ValueError('unable to download %s' % self.name())
#
#     def train_dataset(self, _transforms):
#         dataset = torchvision.datasets.ImageFolder(self.train_path, transform=_transforms)
#         # dataset = torchvision.datasets.CIFAR10(
#         #     root=self.valid_path, train=True, download=False, transform=_transforms)
#         return dataset
#
#     def test_dataset(self, _transforms):
#         dataset = torchvision.datasets.ImageFolder(self.valid_path, transform=_transforms)
#         # dataset = torchvision.datasets.CIFAR10(
#         #     root=self.valid_path, train=False, download=False, transform=_transforms)
#         return dataset
#
#     @property
#     def train_path(self):
#         return os.path.join(self.save_path, 'train_and_valid')
#         # return self.save_path
#
#     @property
#     def valid_path(self):
#         return os.path.join(self.save_path, 'test')
#         # return self.save_path
#
#     @property
#     def normalize(self):
#         return transforms.Normalize(
#             mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
#
#     # def build_train_transform(self, image_size=None, print_log=True):
#     #     if image_size is None:
#     #         image_size = self.image_size
#     #     if print_log:
#     #         print('Color jitter: %s, resize_scale: %s, img_size: %s' %
#     #               (self.distort_color, self.resize_scale, image_size))
#     #
#     #     if self.distort_color == 'torch':
#     #         color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
#     #     elif self.distort_color == 'tf':
#     #         color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
#     #     else:
#     #         color_transform = None
#     #
#     #     if isinstance(image_size, list):
#     #         resize_transform_class = MyRandomResizedCrop
#     #         print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
#     #               'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
#     #     else:
#     #         resize_transform_class = transforms.RandomResizedCrop
#     #
#     #     train_transforms = [
#     #         resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
#     #         transforms.RandomHorizontalFlip(),
#     #     ]
#     #     if color_transform is not None:
#     #         train_transforms.append(color_transform)
#     #     train_transforms += [
#     #         transforms.ToTensor(),
#     #         self.normalize,
#     #     ]
#     #
#     #     train_transforms = transforms.Compose(train_transforms)
#     #     return train_transforms
#
#     def build_train_transform(self, image_size=None, print_log=True, auto_augment='rand-m9-mstd0.5'):
#         if image_size is None:
#             image_size = self.image_size
#         if print_log:
#             print('Color jitter: %s, resize_scale: %s, img_size: %s' %
#                   (self.distort_color, self.resize_scale, image_size))
#
#         if isinstance(image_size, list):
#             resize_transform_class = MyRandomResizedCrop
#             print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
#                   'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
#             img_size_min = min(image_size)
#         else:
#             resize_transform_class = transforms.RandomResizedCrop
#             img_size_min = image_size
#
#         train_transforms = [
#             resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
#             transforms.RandomHorizontalFlip(),
#         ]
#
#         aa_params = dict(
#             translate_const=int(img_size_min * 0.45),
#             img_mean=tuple([min(255, round(255 * x)) for x in [0.47889522, 0.47227842, 0.43047404]]),
#         )
#         aa_params['interpolation'] = _pil_interp('bicubic')
#         train_transforms += [rand_augment_transform(auto_augment, aa_params)]
#
#         train_transforms += [
#             transforms.ToTensor(),
#             Cutout(length=56),
#             self.normalize,
#         ]
#
#         train_transforms = transforms.Compose(train_transforms)
#         return train_transforms
#
#     def build_valid_transform(self, image_size=None):
#         if image_size is None:
#             image_size = self.active_img_size
#         return transforms.Compose([
#             transforms.Resize(int(math.ceil(image_size / 0.875))),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             self.normalize,
#         ])
#
#     def assign_active_img_size(self, new_img_size):
#         self.active_img_size = new_img_size
#         if self.active_img_size not in self._valid_transform_dict:
#             self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
#         # change the transform of the valid and test set
#         self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
#         self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]
#
#     def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
#         # used for resetting running statistics
#         if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
#             if num_worker is None:
#                 num_worker = self.train.num_workers
#
#             n_samples = len(self.train.dataset.samples)
#             g = torch.Generator()
#             g.manual_seed(DataProvider.SUB_SEED)
#             rand_indexes = torch.randperm(n_samples, generator=g).tolist()
#
#             new_train_dataset = self.train_dataset(
#                 self.build_train_transform(image_size=self.active_img_size, print_log=False))
#             chosen_indexes = rand_indexes[:n_images]
#             if num_replicas is not None:
#                 sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
#             else:
#                 sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
#             sub_data_loader = torch.utils.data.DataLoader(
#                 new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
#                 num_workers=num_worker, pin_memory=True,
#             )
#             self.__dict__['sub_train_%d' % self.active_img_size] = []
#             for images, labels in sub_data_loader:
#                 self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
#         return self.__dict__['sub_train_%d' % self.active_img_size]