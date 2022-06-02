from collections import OrderedDict

import torch
import torch.nn as nn

from ofa.utils import MyModule, get_same_padding, build_activation
from ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d
from ofa.utils.layers import set_layer_from_config, ConvLayer, IdentityLayer, LinearLayer

__all__ = ['BasicBlock', 'ResNets', 'ResNet18D']


class BasicBlock(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=1.0, mid_channels=None, act_func='relu', groups=1,
                 downsample_mode='avgpool_conv'):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.groups = groups

        self.downsample_mode = downsample_mode

        if self.mid_channels is None:
            feature_dim = round(self.out_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        self.mid_channels = feature_dim

        # build modules
        pad = get_same_padding(self.kernel_size)
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.in_channels, feature_dim, kernel_size, stride, pad, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, self.out_channels, kernel_size, 1, pad, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(self.out_channels)),
        ]))

        if stride == 1 and in_channels == out_channels:
            self.downsample = IdentityLayer(in_channels, out_channels)
            # self.downsample = None

        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                ('conv', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        # residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        # if self.downsample:
        #     residual = self.downsample(residual)

        x += residual
        x = self.final_act(x)

        return x

    @property
    def module_str(self):
        return '(%s, %s)' % (
            '%dx%d_BottleneckConv_%d->%d->%d_S%d_G%d' % (
                self.kernel_size, self.kernel_size, self.in_channels, self.mid_channels, self.out_channels,
                self.stride, self.groups
            ),
            'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            'name': BasicBlock.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'act_func': self.act_func,
            'groups': self.groups,
            'downsample_mode': self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return BasicBlock(**config)


class ResNets(MyNetwork):
    BASE_DEPTH_LIST = [1, 1, 2, 1]
    STAGE_WIDTH_LIST = [64, 128, 256, 512]

    def __init__(self, input_stem, blocks, classifier, features_only=False):
        super(ResNets, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

        self.features_only = features_only
        self.feature_dim = None

    def forward_features(self, x):
        features = []
        for block in self.blocks:
            y = x
            x = block(x)
            if y.size()[2:] != x.size()[2:]:
                features.append(y)

        features.append(x)
        return features[-3:]  # assumes 1/8, 1/16 and 1/32 features

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        x = self.max_pooling(x)

        if self.features_only:
            return self.forward_features(x)
        else:
            for block in self.blocks:
                x = block(x)
            x = self.global_avg_pool(x)
            x = self.classifier(x)
            return x

    @property
    def module_str(self):
        _str = ''
        for layer in self.input_stem:
            _str += layer.module_str + '\n'
        _str += 'max_pooling(ks=3, stride=2)\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        if self.feature_dim is None:
            # run a dummy forward pass to collect output for measuring feature dimensions
            x = torch.rand(1, 3, 224, 224)
            for layer in self.input_stem:
                x = layer(x)
            x = self.max_pooling(x)
            features = self.forward_features(x)
            self.feature_dim = [v.size(1) for v in features]

        return {
            'name': ResNets.__name__,
            'feature_dim': self.feature_dim,
            'bn': self.get_bn_param(),
            'input_stem': [
                layer.config for layer in self.input_stem
            ],
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        classifier = set_layer_from_config(config['classifier'])

        input_stem = []
        for layer_config in config['input_stem']:
            input_stem.append(set_layer_from_config(layer_config))
        blocks = []
        for block_config in config['blocks']:
            blocks.append(set_layer_from_config(block_config))

        net = ResNets(input_stem, blocks, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, BasicBlock) and isinstance(m.downsample, IdentityLayer):
                m.conv3.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if not isinstance(block.downsample, IdentityLayer) and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        super(ResNets, self).load_state_dict(state_dict)


class ResNet18D(ResNets):

    def __init__(self, n_classes=1000, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0,
                 expand_ratio=None, depth_param=None):

        expand_ratio = 1.0 if expand_ratio is None else expand_ratio

        input_channel = make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        mid_input_channel = make_divisible(input_channel // 2, MyNetwork.CHANNEL_DIVISIBLE)
        stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        depth_list = [2, 2, 2, 2]
        if depth_param is not None:
            for i, depth in enumerate(ResNets.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2, 2]

        # build input stem
        input_stem = [
            ConvLayer(3, mid_input_channel, 3, stride=2, use_bn=True, act_func='relu'),
            ConvLayer(mid_input_channel, mid_input_channel, 3, stride=1, use_bn=True, act_func='relu'),
            ConvLayer(mid_input_channel, input_channel, 3, stride=1, use_bn=True, act_func='relu')
        ]

        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = BasicBlock(
                    input_channel, width, kernel_size=3, stride=stride, expand_ratio=expand_ratio,
                    act_func='relu', downsample_mode='avgpool_conv',
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(ResNet18D, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)

#
# model = ResNet18D()
# print(model)
#
# import torch
# from torchprofile import profile_macs
#
# param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
# param_count_full = sum(p.numel() for p in model.parameters())
#
# print(param_count / 1e6)
# print(param_count_full / 1e6)
#
# data = torch.rand(1, 3, 224, 224)
# model.eval()
# out = model(data)
# # for v in out[1]:
# #     print(v.size())
# flops = profile_macs(model, data) / 1e6
# print(flops)


