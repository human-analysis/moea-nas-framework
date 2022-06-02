import sys
sys.path.insert(0, './')

import copy
import time
import warnings
from abc import ABC
from tqdm import tqdm

import torch

from ofa.utils import get_net_device
from torchprofile import profile_macs
from ofa.imagenet_classification.data_providers.base_provider import DataProvider

from evaluation.evaluate import validate
from supernets.ofa_mbnv3 import GenOFAMobileNetV3
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics


class OFAEvaluator(ABC):
    def __init__(self,
                 supernet: GenOFAMobileNetV3,
                 data_provider: DataProvider,  # data provider class
                 sub_train_size=2000,  # number of images to calibrate BN stats
                 sub_train_batch_size=200,  # batch size for subset train dataloader
                 ):
        self.supernet = supernet
        self.data_provider = data_provider
        self.num_classes = data_provider.n_classes
        self.sub_train_size = sub_train_size
        self.sub_train_batch_size = sub_train_batch_size

        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _calc_params(subnet):
        return sum(p.numel() for p in subnet.parameters() if p.requires_grad) / 1e6  # in unit of Million

    @staticmethod
    def _calc_flops(subnet, dummy_data):
        dummy_data = dummy_data.to(get_net_device(subnet))
        return profile_macs(subnet, dummy_data) / 1e6  # in unit of MFLOPs

    @staticmethod
    def measure_latency(subnet, input_size, iterations=None):
        """ Be aware that latency will fluctuate depending on the hardware operating condition,
        e.g., loading, temperature, etc. """

        print("measuring latency....")

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        subnet.eval()
        model = subnet.cuda()
        input = torch.randn(*input_size).cuda()

        with torch.no_grad():
            for _ in range(10):
                model(input)

            if iterations is None:
                elapsed_time = 0
                iterations = 100
                while elapsed_time < 1:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(iterations):
                        model(input)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t_start
                    iterations *= 2
                FPS = iterations / elapsed_time
                iterations = int(FPS * 6)

            print('=========Speed Testing=========')
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in tqdm(range(iterations)):
                model(input)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            latency = elapsed_time / iterations * 1000
        torch.cuda.empty_cache()
        # FPS = 1000 / latency (in ms)
        return latency

    def _measure_latency(self, subnet, input_size):
        return self.measure_latency(subnet, input_size)

    def eval_acc(self, subnet, dl, sdl, criterion):

        # reset BN running statistics
        subnet.train()
        set_running_statistics(subnet, sdl)
        # measure acc
        loss, (top1, top5) = validate(subnet, dl, criterion)
        return loss, top1, top5

    def evaluate(self, _subnets, objs='acc&flops&params&latency', print_progress=True):
        """ high-fidelity evaluation by inference on validation data """

        subnets = copy.deepcopy(_subnets)  # make a copy of the archs to be evaluated
        batch_stats = []

        for i, subnet_str in enumerate(subnets):
            if print_progress:
                print("evaluating subnet {}:".format(i))
                print(subnet_str)

            stats = {}
            # set subnet accordingly
            image_scale = subnet_str.pop('r')
            input_size = (1, 3, image_scale, image_scale)

            # create dummy data for measuring flops
            dummy_data = torch.rand(*input_size)

            self.supernet.set_active_subnet(**subnet_str)
            subnet = self.supernet.get_active_subnet(preserve_weight=True)
            subnet.cuda()

            print_str = ''
            if 'acc' in objs:
                # set the image scale
                self.data_provider.assign_active_img_size(image_scale)
                dl = self.data_provider.valid
                sdl = self.data_provider.build_sub_train_loader(self.sub_train_size, self.sub_train_batch_size)

                # compute top-1 accuracy
                _, top1, _ = self.eval_acc(subnet, dl, sdl, self.criterion)

                # batch_acc.append(top1)
                stats['acc'] = top1
                print_str += 'Top1 = {:.2f}'.format(top1)

            # calculate #params and #flops
            if 'params' in objs:
                params = self._calc_params(subnet)
                # batch_params.append(params)
                stats['params'] = params
                print_str += ', #Params = {:.2f}M'.format(params)

            if 'flops' in objs:
                with warnings.catch_warnings():  # ignore warnings, use w/ caution
                    warnings.simplefilter("ignore")
                    flops = self._calc_flops(subnet, dummy_data)
                # batch_flops.append(flops)
                stats['flops'] = flops
                print_str += ', #FLOPs = {:.2f}M'.format(flops)

            if 'latency' in objs:
                latency = self._measure_latency(subnet, input_size)
                # batch_latency.append(latency)
                stats['latency'] = latency
                print_str += ', FPS = {:d}'.format(int(1000 / latency))

            if print_progress:
                print(print_str)
            batch_stats.append(stats)

        return batch_stats


class ImageNetEvaluator(OFAEvaluator):
    def __init__(self,
                 supernet: GenOFAMobileNetV3,
                 data_root='../data',  # path to the data folder
                 valid_isze=10000,  # this is a random subset from train used to guide search
                 batchsize=200, n_workers=4,
                 # following two are for BN running stats calibration
                 sub_train_size=2000, sub_train_batch_size=200):

        # build ImageNet dataset and dataloader
        from data_providers.imagenet import ImagenetDataProvider

        imagenet_dataprovider = ImagenetDataProvider(
            save_path=data_root, train_batch_size=batchsize, test_batch_size=batchsize,
            valid_size=valid_isze, n_worker=n_workers)

        super().__init__(supernet, imagenet_dataprovider, num_classes=1000,
                         sub_train_size=sub_train_size, sub_train_batch_size=sub_train_batch_size)


# class CIFAR10Evaluator(OFAEvaluator):
#     def __init__(self,
#                  supernet: GenOFAMobileNetV3,
#                  data_root='../data',  # path to the data folder
#                  valid_isze=2000,  # this is a random subset from train used to guide search
#                  train_batch_size=100, test_batch_size=100, n_workers=4,
#                  # following two are for BN running stats calibration
#                  sub_train_size=960, sub_train_batch_size=96):
#
#         # build CIFAR10 dataset and dataloader
#         from data_providers.cifar import CIFAR10DataProvider
#
#         cifar10_dataprovider = CIFAR10DataProvider(
#             save_path=data_root, train_batch_size=train_batch_size, test_batch_size=test_batch_size,
#             valid_size=valid_isze, n_worker=n_workers)
#
#         super().__init__(supernet, cifar10_dataprovider, num_classes=1000,
#                          sub_train_size=sub_train_size, sub_train_batch_size=sub_train_batch_size)


if __name__ == '__main__':
    from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace

    # construct the supernet
    search_space = OFAMobileNetV3SearchSpace()

    ofa_network = GenOFAMobileNetV3(
        n_classes=1000, dropout_rate=0, image_scale_list=search_space.image_scale_list,
        width_mult_list=search_space.width_mult_list, ks_list=search_space.ks_list,
        expand_ratio_list=search_space.expand_ratio_list, depth_list=search_space.depth_list)

    # load checkpoints weights
    state_dicts = [
        torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.0',
                   map_location='cpu')['state_dict'],
        torch.load('/home/cseadmin/zhichao/neural-architecture-transfer/'
                   'pretrained/backbone/ofa_imagenet/ofa_mbv3_d234_e346_k357_w1.2',
                   map_location='cpu')['state_dict']]

    ofa_network.load_state_dict(state_dicts)

    evaluator = ImageNetEvaluator(
        supernet=ofa_network, data_root='/home/cseadmin/datasets/ILSVRC2012/images', batchsize=200, n_workers=12)

    archs = search_space.sample(5)
    print(archs)
    batch_stats = evaluator.evaluate(archs, objs='acc&flops&params')
    print(archs)
    print(batch_stats)
