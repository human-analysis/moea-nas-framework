""" OFA basic residual block (ResNet18) based supernet"""

import copy
import random
from collections import OrderedDict

import torch.nn as nn

from ofa.utils.layers import set_layer_from_config, IdentityLayer, ResidualBlock
from ofa.utils import make_divisible, val2list, build_activation, get_net_device, MyModule, MyNetwork
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import \
    copy_bn, adjust_bn_according_to_idx, DynamicConvLayer, DynamicLinearLayer
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import \
    DynamicConv2d, DynamicBatchNorm2d, DynamicGroupNorm

from models.resnet18d import ResNets, BasicBlock


__all__ = ['GenOFABasicBlockNets']


class DynamicBasicBlock(MyModule):

    def __init__(self, in_channel_list, out_channel_list, expand_ratio_list=1.0,
                 kernel_size=3, stride=1, act_func='relu', downsample_mode='avgpool_conv'):
        super(DynamicBasicBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel, kernel_size, stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        # self.conv2 = nn.Sequential(OrderedDict([
        #     ('conv', DynamicConv2d(max_middle_channel, max_middle_channel, kernel_size, stride)),
        #     ('bn', DynamicBatchNorm2d(max_middle_channel)),
        #     ('act', build_activation(self.act_func, inplace=True))
        # ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list), kernel_size)),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(max(self.in_channel_list), max(self.out_channel_list))
        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list))),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        # self.conv2.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return '(%s, %s)' % (
            '%dx%d_BasicBlockConv_in->%d->%d_S%d' % (
                self.kernel_size, self.kernel_size, self.active_middle_channels, self.active_out_channel, self.stride
            ),
            'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            'name': DynamicBasicBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'expand_ratio_list': self.expand_ratio_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
            'downsample_mode': self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicBasicBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        layer_config = self.get_active_subnet_config(in_channel)
        if layer_config['name'] == BasicBlock.__name__:
            layer_config.pop('name')
            sub_layer = BasicBlock.build_from_config(layer_config)
        else:
            # sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
            sub_layer = set_layer_from_config(layer_config)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(self.active_middle_channels, in_channel).data)
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        # sub_layer.conv2.conv.weight.data.copy_(
        #     self.conv2.conv.get_active_filter(self.active_middle_channels, self.active_middle_channels).data)
        # copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        # sub_layer.conv3.conv.weight.data.copy_(
        #     self.conv3.conv.get_active_filter(self.active_out_channel, self.active_middle_channels).data)
        # copy_bn(sub_layer.conv3.bn, self.conv3.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(self.active_out_channel, self.active_middle_channels).data)
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(self.active_out_channel, in_channel).data)
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': BasicBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channels,
            'act_func': self.act_func,
            'groups': 1,
            'downsample_mode': self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # # conv3 -> conv2
        # importance = torch.sum(torch.abs(self.conv3.conv.conv.weight.data), dim=(0, 2, 3))
        #
        # if isinstance(self.conv2.bn, DynamicGroupNorm):
        #     channel_per_group = self.conv2.bn.channel_per_group
        #     importance_chunks = torch.split(importance, channel_per_group)
        #     for chunk in importance_chunks:
        #         chunk.data.fill_(torch.mean(chunk))
        #     importance = torch.cat(importance_chunks, dim=0)
        #
        # if expand_ratio_stage > 0:
        #     sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
        #     sorted_expand_list.sort(reverse=True)
        #     target_width_list = [
        #         make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
        #         for expand in sorted_expand_list
        #     ]
        #     right = len(importance)
        #     base = - len(target_width_list) * 1e5
        #     for i in range(expand_ratio_stage + 1):
        #         left = target_width_list[i]
        #         importance[left:right] += base
        #         base += 1e5
        #         right = left
        #
        # sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        # self.conv3.conv.conv.weight.data = torch.index_select(self.conv3.conv.conv.weight.data, 1, sorted_idx)
        # adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        # self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 0, sorted_idx)

        # conv2 -> conv1
        importance = torch.sum(torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.conv1.bn, DynamicGroupNorm):
            channel_per_group = self.conv1.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(self.conv1.conv.conv.weight.data, 0, sorted_idx)

        return None


class GenOFABasicBlockNets(ResNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
                 depth_list=0, expand_ratio_list=1.0, width_mult_list=1.0, features_only=False):

        self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        # self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE) for channel in input_channel
        ]

        stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
            ]

        n_block_list = [base_depth + depth for depth, base_depth in
                        zip(self.depth_list, ResNets.BASE_DEPTH_LIST)]
        stride_list = [1, 2, 2, 2]

        # build input stem
        input_stem = [
            DynamicConvLayer(val2list(3), mid_input_channel, 3, stride=2, use_bn=True, act_func='relu'),
            DynamicConvLayer(mid_input_channel, mid_input_channel, 3, stride=1, use_bn=True, act_func='relu'),
            DynamicConvLayer(mid_input_channel, input_channel, 3, stride=1, use_bn=True, act_func='relu')
        ]

        # blocks
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = DynamicBasicBlock(
                    input_channel, width, expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3, stride=stride, act_func='relu', downsample_mode='avgpool_conv',
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = DynamicLinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(GenOFABasicBlockNets, self).__init__(input_stem, blocks, classifier)

        self.features_only = features_only

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return 'OFABasicBlockNets'

    def forward_features(self, x):
        features = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)

            # assumes 1/8, 1/16 and 1/32 features
            if stage_id > 0:
                features.append(x)
        return features

    def forward(self, x):
        for layer in self.input_stem:
            # if self.input_stem_skipping > 0 \
            #         and isinstance(layer, ResidualBlock) and isinstance(layer.shortcut, IdentityLayer):
            #     pass
            # else:
            x = layer(x)
        x = self.max_pooling(x)
        if self.features_only:
            return self.forward_features(x)
        else:
            for stage_id, block_idx in enumerate(self.grouped_block_index):
                depth_param = self.runtime_depth[stage_id]
                active_idx = block_idx[:len(block_idx) - depth_param]
                for idx in active_idx:
                    x = self.blocks[idx](x)
            x = self.global_avg_pool(x)
            x = self.classifier(x)
            return x

    @property
    def module_str(self):
        _str = ''
        for layer in self.input_stem:
            if self.input_stem_skipping > 0 \
                    and isinstance(layer, ResidualBlock) and isinstance(layer.shortcut, IdentityLayer):
                pass
            else:
                _str += layer.module_str + '\n'
        _str += 'max_pooling(ks=3, stride=2)\n'
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': GenOFABasicBlockNets.__name__,
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
        raise ValueError('do not support this function')

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key
            if new_key in model_dict:
                pass
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(GenOFABasicBlockNets, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    # def set_max_net(self):
    #     self.set_active_subnet(d=max(self.depth_list), e=max(self.expand_ratio_list), w=len(self.width_mult_list) - 1)

    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        # depth = val2list(d, len(ResNets.BASE_DEPTH_LIST) + 1)
        depth = val2list(d, len(ResNets.BASE_DEPTH_LIST))
        expand_ratio = val2list(e, len(self.blocks))
        # width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST) + 2)
        width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST))

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        # if width_mult[0] is not None:
        #     self.input_stem[1].conv.active_out_channel = self.input_stem[0].active_out_channel = \
        #         self.input_stem[0].out_channel_list[width_mult[0]]
        # if width_mult[1] is not None:
        #     self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[width_mult[1]]

        if width_mult[0] is not None:
            self.input_stem[1].active_out_channel = self.input_stem[0].active_out_channel = \
                self.input_stem[0].out_channel_list[width_mult[0]]
            self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[width_mult[1]]
        # if width_mult[1] is not None:
        #     self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[width_mult[1]]

        # if depth[0] is not None:
        #     self.input_stem_skipping = (depth[0] != max(self.depth_list))

        # for stage_id, (block_idx, d, w) in enumerate(zip(self.grouped_block_index, depth[1:], width_mult[2:])):
        for stage_id, (block_idx, d, w) in enumerate(zip(self.grouped_block_index, depth, width_mult[1:])):
            if d is not None:
                # self.runtime_depth[stage_id] = max(self.depth_list) - d
                self.runtime_depth[stage_id] = self.depth_list[stage_id] - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[idx].out_channel_list[w]

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_lb_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_width_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_lb_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_width_include_list'] = None

    def sample_active_subnet(self):

        # for width mult, we need to convert from actual wd values to wd choices, i.e., [0.65, 0.8, 1.0] --> [0, 1, 2]
        width_candidates = [0, 1, 2] if self.__dict__.get('_width_include_list', None) is None \
            else self.__dict__['_width_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_lb = [0, 0, 0, 0] if self.__dict__.get('_depth_lb_list', None) is None else \
            self.__dict__['_depth_lb_list']

        # print(width_candidates)
        # print(expand_candidates)
        # print(depth_lb)

        # sample expand ratio
        expand_setting = []
        # for block in self.blocks:
            # expand_setting.append(random.choice(block.expand_ratio_list))
        for _ in self.blocks:
            expand_setting.append(random.choice(expand_candidates))

        # sample depth
        # depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]
        depth_setting = []
        for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
            # depth_setting.append(random.choice(self.depth_list))
            depth_setting.append(random.choice(range(depth_lb[stage_id], self.depth_list[stage_id] + 1)))

        # sample width_mult
        width_mult_setting = [
            # random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
            # random.choice(list(range(len(self.input_stem[2].out_channel_list)))),
            random.choice(width_candidates),
        ]
        # for stage_id, block_idx in enumerate(self.grouped_block_index):
            # stage_first_block = self.blocks[block_idx[0]]
            # width_mult_setting.append(
            #     random.choice(list(range(len(stage_first_block.out_channel_list))))
            # )
        for _ in range(len(self.grouped_block_index)):
            width_mult_setting.append(random.choice(width_candidates))

        arch_config = {
            'd': depth_setting,
            'e': expand_setting,
            'w': width_mult_setting
        }

        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_subnet(self, preserve_weight=True):

        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight),
                      self.input_stem[1].get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight),
                      self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight)]

        # if self.input_stem_skipping <= 0:
        #     input_stem.append(ResidualBlock(
        #         self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight),
        #         IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
        #     ))

        input_channel = self.input_stem[2].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(self.blocks[idx].get_active_subnet(input_channel, preserve_weight))
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        # subnet = ResNets(input_stem, blocks, classifier)
        subnet = ResNets(input_stem, blocks, classifier, features_only=self.features_only)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3),
                             self.input_stem[1].get_active_subnet_config(self.input_stem[0].active_out_channel),
                             self.input_stem[2].get_active_subnet_config(self.input_stem[0].active_out_channel)]
        # if self.input_stem_skipping <= 0:
        #     input_stem_config.append({
        #         'name': ResidualBlock.__name__,
        #         'conv': self.input_stem[1].conv.get_active_subnet_config(self.input_stem[0].active_out_channel),
        #         'shortcut': IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel),
        #     })
        input_channel = self.input_stem[2].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(self.blocks[idx].get_active_subnet_config(input_channel))
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            'name': ResNets.__name__,
            'bn': self.get_bn_param(),
            'input_stem': input_stem_config,
            'blocks': blocks_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)


if __name__ == '__main__':
    import torch

    data = torch.rand(1, 3, 224, 224)

    # # inverted bottleneck block ofa backbone
    ofa_network = OFABasicBlockNets(
        dropout_rate=0, width_mult_list=[0.65, 0.8, 1.0], expand_ratio_list=[0.65, 0.8, 1.0], depth_list=[2, 3, 4, 2],
        features_only=True)

    # # set subnet sampling constraints
    # ofa_network.set_constraint([1, 2, 2, 1], constraint_type='depth')
    # ofa_network.set_constraint([0.8, 1.0], constraint_type='expand_ratio')
    # ofa_network.set_constraint([1, 2], constraint_type='width_mult')
    # # clear all subnet sampling constraints
    # ofa_network.clear_constraint()
    # # sample
    # arch_config = ofa_network.sample_active_subnet()
    # print(arch_config)

    # set subnet directly
    ofa_network.set_active_subnet(d=[0, 0, 0, 0], e=[0.65] * 16, w=[1] * 5)

    # Manually setting sub-network to minimal bound subnet
    # ofa_network.set_active_subnet(d=[0, 0, 0, 0], e=[1.0] * 16, w=[0] * 5)
    # print(arch_config)

    subnet = ofa_network.get_active_subnet()
    # print(subnet)
    out = subnet(data)
    if type(out) == list:
        print([v.size() for v in out])
    else:
        print(out.size())

    print(subnet.config['feature_dim'])
    # flops = profile_macs(ofa_network, data) / 1e6
    # print(flops)

    # Randomly sample sub-networks from OFA network
    # random_subnet = ofa_network.get_active_subnet(preserve_weight=True)
    # print(random_subnet)

    # weights = torch.load("pretrained/resnet12d.pth", map_location='cpu')
    # random_subnet.load_state_dict(weights)
    #
    # random_subnet.eval()
    # out = random_subnet(data)

    # if type(out) == list:
    #     print([v.size() for v in out])
    # else:
    #     print(out.size())
    # print(random_subnet.config)

    # # for v in out[1]:
    # #     print(v.size())
    # flops = profile_macs(random_subnet, data) / 1e6
    # print(flops)