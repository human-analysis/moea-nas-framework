import sys
from copy import deepcopy

sys.path.insert(0, './')

import torch

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicConvLayer

from models.fanet import GenFANet, ContextPath, FeatureAlign, Context, BiSeNetOutput, FeatureSelectionModule


class OFAFANet(GenFANet):
    def __init__(self,
                 backbone_option='basic',
                 depth_list=None,
                 expand_ratio_list=None,
                 width_mult_list=None,
                 feature_dim_list=None,
                 n_classes=19, out_nc=128, output_aux=False
                 ):

        if backbone_option == 'basic':
            from supernets.ofa_basic import GenOFABasicBlockNets
            # depth_list = [2, 3, 4, 2]; expand_ratio_list = [0.65, 0.8, 1.0]; width_mult_list = [0.65, 0.8, 1.0];
            # feature_dim_list = [[80, 104, 128], [168, 208, 256], [336, 408, 512]]
            backbone = GenOFABasicBlockNets(
                dropout_rate=0, depth_list=depth_list, expand_ratio_list=expand_ratio_list,
                width_mult_list=width_mult_list, features_only=True)

        # elif backbone_option == 'bottleneck':
        #     from supernet.ofa_bottleneck import OFABottleneckNets
        #     # depth_list = [0, 1, 2]; expand_ratio_list = [0.2, 0.25, 0.35]; width_mult_list = [0.65, 0.8, 1.0];
        #     # feature_dim_list = [[80, 104, 128], [168, 208, 256], [336, 408, 512]]
        #     backbone = OFABottleneckNets(
        #         dropout_rate=0, depth_list=depth_list, expand_ratio_list=expand_ratio_list,
        #         width_mult_list=width_mult_list, features_only=True)
        #
        # elif backbone_option == 'inverted':
        #     from supernet.ofa_inverted import OFAInvertedBottleneckNets
        #     # depth_list = [2, 3, 4]; expand_ratio_list = [3, 4, 6]; width_mult_list = 1.3/1.0;
        #     # feature_dim_list = [[56], [128], [416]]
        #     backbone = OFAInvertedBottleneckNets(
        #         dropout_rate=0, width_mult=width_mult_list, ks_list=3, expand_ratio_list=expand_ratio_list,
        #         depth_list=depth_list, features_only=True)

        else:
            raise NotImplementedError

        nc_8_list, nc_16_list, nc_32_list = feature_dim_list

        cp = ContextPath(backbone)

        context = Context(FeatureSelectionModule(DynamicConvLayer(
            nc_32_list, [out_nc], kernel_size=1, stride=1, use_bn=True, act_func='relu')))

        align16 = FeatureAlign(FeatureSelectionModule(DynamicConvLayer(
            nc_16_list, [out_nc], kernel_size=1, stride=1, use_bn=True, act_func='relu')), out_nc=out_nc)

        align8 = FeatureAlign(FeatureSelectionModule(DynamicConvLayer(
            nc_8_list, [out_nc], kernel_size=1, stride=1, use_bn=True, act_func='relu')), out_nc=out_nc)

        conv_out = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out16 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out32 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)

        super(OFAFANet, self).__init__(cp, context, align16, align8, conv_out32, conv_out16, conv_out,
                                       out_nc, n_classes, output_aux)

        self.out_nc = out_nc
        # self.backbone = backbone

    @staticmethod
    def name():
        return 'OFAFANet'

    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        if (d is not None) and (e is not None) and (w is not None):
            self.cp.backbone.set_active_subnet(d=d, e=e, w=w, **kwargs)
        self.context.fsm.conv.active_out_channel = self.out_nc
        self.align16.fsm.conv.active_out_channel = self.out_nc
        self.align8.fsm.conv.active_out_channel = self.out_nc

    def sample_active_subnet(self):
        backbone_config = self.cp.backbone.sample_active_subnet()  # this line sets the backbone active subnet as well
        self.set_active_subnet()
        return backbone_config

    def get_active_subnet(self, preserve_weight=True, output_aux=False):
        backbone = self.cp.backbone.get_active_subnet(preserve_weight)
        nc_8, nc_16, nc_32 = backbone.config['feature_dim']

        cp = ContextPath(backbone)
        context = deepcopy(self.context)
        context.fsm.conv = context.fsm.conv.get_active_subnet(nc_32, preserve_weight)

        align16 = deepcopy(self.align16)
        align16.fsm.conv = align16.fsm.conv.get_active_subnet(nc_16, preserve_weight)

        align8 = deepcopy(self.align8)
        align8.fsm.conv = align8.fsm.conv.get_active_subnet(nc_8, preserve_weight)

        # self.context.fsm = self.context.fsm.conv.get_active_subnet(nc_32, preserve_weight)
        # self.align16.fsm = self.align16.fsm.conv.get_active_subnet(nc_16, preserve_weight)
        # self.align8.fsm = self.align8.fsm.conv.get_active_subnet(nc_8, preserve_weight)

        subnet = GenFANet(cp, context, align16, align8, out8=deepcopy(self.conv_out),
                          out16=deepcopy(self.conv_out16), out32=deepcopy(self.conv_out32), output_aux=output_aux)

        return subnet

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)

    def load_backbone_state_dict(self, pth2weights):
        state_dict = torch.load(pth2weights, map_location='cpu')
        try:
            self.cp.backbone.load_state_dict(state_dict)
        except ValueError:
            self.cp.backbone.load_state_dict(state_dict['state_dict'])
        print("Backbone weights loaded.")


if __name__ == '__main__':

    data = torch.rand(1, 3, 512, 1024)
    data = data.cuda()
    # print(data)
    # ------------------------------ basic block ------------------------------ #
    # ofa_network = OFAFANet(
    #     backbone_option='basic', depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0],
    #     width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]])

    # load backbone weights
    # ofa_network.load_backbone_state_dict("pretrained/backbone/basic.pth")

    # load fanet weights
    # state_dict = torch.load("pretrained/ofa_fanet34d/model_maxmIOU50.pth", map_location='cpu')
    # ofa_network.load_state_dict(state_dict)
    #
    # ofa_network.cp.backbone.set_constraint([1, 2, 2, 1], constraint_type='depth')
    # ofa_network.cp.backbone.set_constraint([0.8, 1.0], constraint_type='expand_ratio')
    # ofa_network.cp.backbone.set_constraint([1, 2], constraint_type='width_mult')

    # ------------------------------ inverted bottleneck block ------------------------------ #
    ofa_network = OFAFANet(
        backbone_option='inverted', depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6],
        width_mult_list=1.0, feature_dim_list=[[40], [96], [320]])

    # load backbone weights
    ofa_network.load_backbone_state_dict("pretrained/backbone/inverted.pth")
    ofa_network.cuda()
    # bottleneck block
    # ofa_network = OFAFANet(
    #     backbone_option='bottleneck', depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35],
    #     width_mult_list=[0.65, 0.8, 1.0], feature_dim_list=[[80, 104, 128], [168, 208, 256], [336, 408, 512]])
    #
    # # inverted bottleneck block
    # ofa_network = OFAFANet(
    #     backbone_option='inverted', depth_list=[2, 3, 4], expand_ratio_list=[3, 4, 6], width_mult_list=1.3,
    #     feature_dim_list=[[56], [128], [416]])
    # ofa_network.load_backbone_state_dict("pretrained/backbone/inverted_bottleneck.pth")
    # print(ofa_network)

    backbone_config = ofa_network.sample_active_subnet()
    print(backbone_config)
    # subnet = ofa_network.get_active_subnet(preserve_weight=True)
    # subnet.cuda()
    # print(subnet)

    # subnet.eval()
    out = ofa_network(data)
    print(out[0].size())