#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from tqdm import tqdm

from ofa.utils import get_net_device

from evaluation.utils import *


def validate(net, data_loader, criterion, epoch=0, run_str='', no_logs=False, train_mode=False):

    if train_mode:
        net.train()
    else:
        net.eval()

    device = get_net_device(net)

    losses = AverageMeter()
    metric_dict = get_metric_dict()

    with torch.no_grad():
        with tqdm(total=len(data_loader),
                  desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                # compute output
                output = net(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                update_metric(metric_dict, output, labels)

                losses.update(loss.item(), images.size(0))
                t.set_postfix({
                    'loss': losses.avg,
                    **get_metric_vals(metric_dict, return_dict=True),
                    'img_size': images.size(2),
                })
                t.update(1)

    return losses.avg, get_metric_vals(metric_dict)



