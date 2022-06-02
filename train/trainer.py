import sys
sys.path.insert(0, './')

import os
import torch
import random
import itertools
import numpy as np
from abc import ABC
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler
from cutmix.utils import CutMixCrossEntropyLoss
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.imagenet_classification.data_providers.base_provider import DataProvider
from ofa.utils import cross_entropy_loss_with_soft_target, MyRandomResizedCrop, subset_mean, list_mean

from evaluation.evaluate import validate
from search.search_spaces import SearchSpace
from supernets.ofa_mbnv3 import GenOFAMobileNetV3
from train.utils import ModelEma, AverageMeter, create_exp_dir, get_state_dict

# figure out if the model can be trained with mixed-precision
from contextlib import suppress
from timm.utils import ApexScaler, NativeScaler
try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class Trainer(ABC):
    def __init__(self,
                 model,  # the network model to be trained
                 # below are training configurations
                 n_epochs,  # number of training epochs
                 lr_init,  # initial learning rate
                 data_provider: DataProvider,  # data provider including both train and valid data loader
                 cur_epoch=0,  # current epoch from which to train
                 lr_end=0.,  # final learning rate
                 lr_warmup=0.0001,  # warmup learning rate
                 lr_warmup_epochs=5,  # number of epochs to warm-up learning rate
                 momentum=0.9,
                 wd=3e-4,  # weight decay
                 grad_clip=5,  # gradient clipping
                 model_ema=False,  # keep track of moving average of model parameters
                 model_ema_decay=0.9998,  # decay factor for model weights moving average
                 logger=None,  # logger handle for recording
                 save='.tmp'  # path to save experiment data
                 ):

        self.cur_epoch = cur_epoch
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip

        self.data_provider = data_provider
        # self.train_dataloader = train_dataloader
        # self.valid_dataloader = valid_dataloader

        self.optimizer = torch.optim.SGD(model.parameters(), lr_init, momentum=momentum, weight_decay=wd)
        self.criterion = CutMixCrossEntropyLoss()  # assuming using cutmix data augmentation
        lr_cycle_args = {'cycle_mul': 1., 'cycle_decay': 0.1, 'cycle_limit': 1}
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=n_epochs,
            lr_min=lr_end,
            warmup_lr_init=lr_warmup,
            warmup_t=lr_warmup_epochs,
            k_decay=1.0,
            **lr_cycle_args,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs, eta_min=lr_end)

        self.model = model

        if model_ema:
            self.model_ema = ModelEma(model, decay=model_ema_decay)
        else:
            self.model_ema = model_ema

        self.save = save
        create_exp_dir(self.save)  # create an experiment folder

        # setup progress logger
        if logger is None:
            import sys
            import logging

            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            self.logger = logging.getLogger()
            self.logger.addHandler(fh)
        else:
            self.logger = logger

        # resolve AMP arguments based on PyTorch / Apex availability
        use_amp = None
        if has_apex:
            use_amp = 'apex'
        elif has_native_amp:
            use_amp = 'native'
        else:
            print("Neither APEX or native Torch AMP is available, using float32. "
                  "Install NVIDA apex or upgrade to PyTorch 1.6")

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        if use_amp == 'apex':
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.loss_scaler = ApexScaler()
            self.logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif use_amp == 'native':
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            self.logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            self.logger.info('AMP not enabled. Training in float32.')

    def train_one_epoch(self, epoch, run_str='', no_logs=False):
        self.model.train()

        MyRandomResizedCrop.EPOCH = epoch  # such that sampled image resolution is changing
        n_batch = len(self.data_provider.train)
        losses = AverageMeter()

        with tqdm(total=len(self.data_provider.train),
                  desc='Training Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
            num_updates = epoch * n_batch
            for step, (x, target) in enumerate(self.data_provider.train):

                MyRandomResizedCrop.BATCH = step
                x, target = x.cuda(), target.cuda(non_blocking=True)

                with self.amp_autocast():
                    logits = self.model(x)
                    loss = self.criterion(logits, target)

                losses.update(loss.item(), x.size(0))

                self.optimizer.zero_grad()
                if self.loss_scaler is not None:
                    self.loss_scaler(
                        loss, self.optimizer,
                        clip_grad=self.grad_clip, clip_mode='norm',
                        parameters=self.model.parameters(),
                        create_graph=False)
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                if self.model_ema:
                    self.model_ema.update(self.model)

                torch.cuda.synchronize()
                num_updates += 1

                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                self.scheduler.step_update(num_updates=num_updates, metric=losses.avg)

                t.set_postfix({
                    'loss': losses.avg,
                    'lr': lr,
                    'img_size': x.size(2),
                })
                t.update(1)

        return losses.avg

    def validate(self, epoch):
        image_sizes = self.data_provider.image_size
        if not isinstance(image_sizes, list):
            image_sizes = list(image_sizes)

        losses, top1s, top5s = [], [], []
        for img_size in image_sizes:

            # set image size
            self.data_provider.assign_active_img_size(img_size)

            # measure acc
            loss, (top1, top5) = validate(self.model, self.data_provider.test, self.criterion, epoch=epoch)
            losses.append(loss)
            top1s.append(top1)
            top5s.append(top5)

        return np.mean(losses), (np.mean(top1s), np.mean(top5s))

    def train(self):

        for epoch in range(self.cur_epoch, self.n_epochs):
            # self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.scheduler.get_lr()[0]))
            # self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.scheduler.get_epoch_values(epoch)[0]))

            train_loss = self.train_one_epoch(epoch)

            print_str = "train loss = {:.4f}".format(train_loss)

            save_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                         'optimizer_state_dict': self.optimizer.state_dict()}

            # # uncomment below for debugging, cost additional 20 secs per epoch
            # if self.model_ema:
            #     valid_loss_ema, (valid_acc_ema, _) = validate(
            #         self.model_ema.ema, self.valid_dataloader, self.criterion, epoch=epoch)
            #     print_str += ", valid acc ema = {:.4f}".format(valid_acc_ema)
            #     save_dict['state_dict_ema'] = get_state_dict(self.model_ema)

            valid_loss, (valid_acc, _) = self.validate(epoch)
            print_str += ", valid acc = {:.4f}".format(valid_acc)

            self.logger.info(print_str)

            self.scheduler.step(epoch + 1)

            torch.save(save_dict, os.path.join(self.save, 'checkpoint.pth.tar'))


class SuperNetTrainer(ABC):
    def __init__(self,
                 supernet: GenOFAMobileNetV3,  # the supernet model to be trained
                 teacher_model,  # we use the full capacity supernet to supervise the subnet training
                 search_space: SearchSpace,  # the search space from which a subpart of supernet can be sampled
                 # below are training configurations
                 n_epochs,  # number of training epochs
                 lr_init,  # initial learning rate
                 data_provider: DataProvider,  # data provider including both train and valid data loader
                 cur_epoch=0,  # current epoch from which to train
                 lr_end=0.,  # final learning rate
                 lr_warmup=0.0001,  # warmup learning rate
                 lr_warmup_epochs=5,  # number of epochs to warm-up learning rate
                 momentum=0.9,
                 wd=3e-4,  # weight decay
                 grad_clip=5,  # gradient clipping
                 supernet_ema=False,  # keep track of moving average of model parameters
                 supernet_ema_decay=0.9998,  # decay factor for model weights moving average
                 logger=None,  # logger handle for recording
                 save='.tmp',  # path to save experiment data
                 # additional arguments for training supernet
                 dynamic_batch_size=1,  # number of architectures to accumulate gradient
                 kd_ratio=1.0,  # teacher-student knowledge distillation hyperparameter
                 sub_train_size=2000,  # number of images to calibrate BN stats
                 sub_train_batch_size=200,  # batch size for subset train dataloader
                 distributions=None,  # instead of uniform sampling from search space,
                                      # we can sample explicitly from a distribution
                 report_freq=10,  # frequency to run validation and print results
                 ):

        self.cur_epoch = cur_epoch
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip
        self.report_freq = report_freq

        self.data_provider = data_provider
        self.sub_train_size = sub_train_size
        self.sub_train_batch_size = sub_train_batch_size

        self.distributions = distributions

        self.optimizers = [torch.optim.SGD(model.parameters(), lr_init, momentum=momentum, weight_decay=wd)
                           for model in supernet.engine]  # we have one optimizer for supernet of each wid_mult
        lr_cycle_args = {'cycle_mul': 1., 'cycle_decay': 0.1, 'cycle_limit': 1}
        self.schedulers = [CosineLRScheduler(
            optimizer,
            t_initial=n_epochs,
            lr_min=lr_end,
            warmup_lr_init=lr_warmup,
            warmup_t=lr_warmup_epochs,
            k_decay=1.0,
            **lr_cycle_args,
        ) for optimizer in self.optimizers]
        # self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=lr_end)
        #                    for optimizer in self.optimizers]  # we have one lr scheduler for supernet of each wid_mult
        self.criterion = CutMixCrossEntropyLoss()  # assuming using cutmix data augmentation

        self.supernet = supernet

        if supernet_ema:
            self.supernet_ema = [ModelEma(model, decay=supernet_ema_decay) for model in self.supernet.engine]
        else:
            self.supernet_ema = supernet_ema

        self.teacher_model = teacher_model
        self.search_space = search_space
        self.dynamic_batch_size = dynamic_batch_size
        self.kd_ratio = kd_ratio

        self.save = save
        create_exp_dir(self.save)  # create an experiment folder

        # setup progress logger
        if logger is None:
            import sys
            import logging

            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            self.logger = logging.getLogger()
            self.logger.addHandler(fh)
        else:
            self.logger = logger

        # # resolve AMP arguments based on PyTorch / Apex availability
        # use_amp = None
        # if has_apex:
        #     use_amp = 'apex'
        # elif has_native_amp:
        #     use_amp = 'native'
        # else:
        #     print("Neither APEX or native Torch AMP is available, using float32. "
        #           "Install NVIDA apex or upgrade to PyTorch 1.6")

        # # setup automatic mixed-precision (AMP) loss scaling and op casting
        # self.amp_autocast = suppress  # do nothing
        # self.loss_scaler = None
        # if use_amp == 'apex':
        #     for i in range(len(self.optimizers)):
        #         self.supernet.engine[i], self.optimizers[i] = amp.initialize(
        #             self.supernet.engine[i], self.optimizers[i], opt_level='O1')
        #     # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        #     self.loss_scalers = [ApexScaler() for _ in range(len(self.optimizers))]
        #     self.logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        # elif use_amp == 'native':
        #     self.amp_autocast = torch.cuda.amp.autocast
        #     self.loss_scalers = [NativeScaler() for _ in range(len(self.optimizers))]
        #     # self.loss_scaler = NativeScaler()
        #     self.logger.info('Using native Torch AMP. Training in mixed precision.')
        # else:
        #     self.logger.info('AMP not enabled. Training in float32.')

    def train_one_epoch(self, epoch, run_str='', no_logs=False):

        # switch to training mode
        for model in self.supernet.engine:
            model.train()
        # self.model.train()

        MyRandomResizedCrop.EPOCH = epoch  # such that sampled image resolution is changing

        n_batch = len(self.data_provider.train)

        losses = AverageMeter()

        with tqdm(total=n_batch, desc='Training Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:

            num_updates = epoch * n_batch
            for step, (x, target) in enumerate(self.data_provider.train):

                MyRandomResizedCrop.BATCH = step
                x, target = x.cuda(), target.cuda(non_blocking=True)

                # soft target
                with torch.no_grad():
                    soft_logits = self.teacher_model(x).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

                # clean gradients
                for model in self.supernet.engine:
                    model.zero_grad()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                # sample architectures from search space
                subnets = self.search_space.sample(self.dynamic_batch_size, distributions=self.distributions)

                subnet_str, loss_of_subnets = '', []
                for _, subnet_settings in enumerate(subnets):
                    # set random seed before sampling
                    subnet_seed = int('%d%.3d%.3d' % (epoch * n_batch + step, _, 0))
                    random.seed(subnet_seed)
                    subnet_settings.pop('r')  # remove image size key
                    self.supernet.set_active_subnet(**subnet_settings)
                    subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()]) + ' || '

                    # with self.amp_autocast():
                    logits = self.supernet.forward(x)
                    kd_loss = cross_entropy_loss_with_soft_target(logits, soft_label)
                    loss = self.kd_ratio * kd_loss + self.criterion(logits, target)
                    loss = loss * (2 / (self.kd_ratio + 1))

                    loss_of_subnets.append(loss.item())

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)

                for optimizer in self.optimizers:
                    optimizer.step()

                if self.supernet_ema:
                    for model, model_ema in zip(self.supernet.engine, self.supernet_ema):
                        model_ema.update(model)

                torch.cuda.synchronize()
                num_updates += 1
                losses.update(list_mean(loss_of_subnets), x.size(0))

                lrl = [param_group['lr'] for param_group in self.optimizers[0].param_groups]
                lr = sum(lrl) / len(lrl)

                for scheduler in self.schedulers:
                    scheduler.step_update(num_updates=num_updates, metric=losses.avg)

                t.set_postfix({
                    'loss': losses.avg,
                    'lr': lr,
                    'img_size': x.size(2),
                    'str': subnet_str,
                })
                t.update(1)

        return losses.avg

    def validate(self, epoch):
        image_sizes = self.data_provider.image_size
        if not isinstance(image_sizes, list):
            image_sizes = list(image_sizes)

        # create the combinations to validate performance
        wid_mult_options = [min(self.supernet.width_mult_list), max(self.supernet.width_mult_list)]
        ks_options = [min(self.supernet.ks_list), max(self.supernet.ks_list)]
        expand_ratio_options = [min(self.supernet.expand_ratio_list), max(self.supernet.expand_ratio_list)]
        depth_options = [min(self.supernet.depth_list), max(self.supernet.depth_list)]
        val_settings = list(itertools.product(
            image_sizes, wid_mult_options, ks_options, expand_ratio_options, depth_options))

        losses, top1s, top5s = [], [], []
        for r, w, ks, e, d in val_settings:

            self.logger.info('r={:d}, w={:.1f}, ks={:d}, e={:d}, d={:d}'.format(r, w, ks, e, d))

            # set image size
            self.data_provider.assign_active_img_size(r)
            dl = self.data_provider.test
            sdl = self.data_provider.build_sub_train_loader(self.sub_train_size, self.sub_train_batch_size)

            # set subnet settings
            self.supernet.set_active_subnet(w=w, ks=ks, e=e, d=d)
            subnet = self.supernet.get_active_subnet(preserve_weight=True)

            # reset BN running statistics
            subnet.train()
            set_running_statistics(subnet, sdl)

            # measure acc
            loss, (top1, top5) = validate(subnet, dl, self.criterion, epoch=epoch)
            losses.append(loss)
            top1s.append(top1)
            top5s.append(top5)

        return np.mean(losses), (np.mean(top1s), np.mean(top5s))

    def train(self):

        for epoch in range(self.cur_epoch, self.n_epochs):

            # self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.schedulers[0].get_lr()[0]))
            self.logger.info('epoch {:d} lr {:.2e}'.format(epoch, self.schedulers[0].get_epoch_values(epoch)[0]))

            train_loss = self.train_one_epoch(epoch)

            print_str = "train loss = {:.4f}".format(train_loss)

            save_dict = {'epoch': epoch, 'model_w1.0_state_dict': self.supernet.engine[0].state_dict(),
                         'model_w1.2_state_dict': self.supernet.engine[1].state_dict(),
                         'optimizer_w1.0_state_dict': self.optimizers[0].state_dict(),
                         'optimizer_w1.2_state_dict': self.optimizers[1].state_dict()}

            if (epoch + 1) % self.report_freq == 0:
                valid_loss, (valid_acc, _) = self.validate(epoch)
                print_str += ", valid acc = {:.4f}".format(valid_acc)

            self.logger.info(print_str)

            for scheduler in self.schedulers:
                scheduler.step(epoch + 1)

            torch.save(save_dict, os.path.join(self.save, 'checkpoint.pth.tar'))


if __name__ == '__main__':
    from supernets.utils import reset_classifier
    from supernets.ofa_mbnv3 import GenOFAMobileNetV3
    from data_providers.cifar import CIFAR10DataProvider
    from search.search_spaces.ofa_search_space import OFAMobileNetV3SearchSpace

    dataprovider = CIFAR10DataProvider(
        save_path='/home/cseadmin/datasets', train_batch_size=96, test_batch_size=200, valid_size=None, n_worker=2,
        image_size=256)
    # MyRandomResizedCrop.CONTINUOUS = True
    # MyRandomResizedCrop.SYNC_DISTRIBUTED = True

    # dataprovider = CIFAR100DataProvider(
    #     save_path='/home/cseadmin/datasets', train_batch_size=96, test_batch_size=200, valid_size=None, n_worker=2,
    #     image_size=256)

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

    arch = search_space.decode([np.array(search_space.ub)])[0]
    print(arch)
    # set the image scale
    image_scale = arch.pop('r')
    dataprovider.assign_active_img_size(image_scale)
    # arch = search_space.sample(1)[0]

    ofa_network.set_active_subnet(**arch)
    model = ofa_network.get_active_subnet(preserve_weight=True)

    reset_classifier(model, n_classes=dataprovider.n_classes)
    model = model.cuda()

    trainer = Trainer(
        model, n_epochs=50, lr_init=0.0025, wd=3e-5, train_dataloader=dataprovider.train,
        valid_dataloader=dataprovider.test, model_ema=False, save='tmp/cifar10')

    trainer.train()