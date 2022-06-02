import torch
import torch.nn as nn
import torch.nn.functional as F

from DCNv2.dcn_v2 import DCN
from timm import create_model
from inplace_abn import InPlaceABNSync as BatchNorm2d  # pip install inplace-abn

from supernets.ofa_basic import GenOFABasicBlockNets


__all__ = ['FANet', 'FeatureAlign', 'ContextPath', 'Context', 'BiSeNetOutput', 'FeatureSelectionModule']


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_chan, activation='identity')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureSelectionModule(nn.Module):
    # def __init__(self, in_chan, out_chan):
    def __init__(self, conv):
        super(FeatureSelectionModule, self).__init__()
        # self.conv = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv = conv

    def forward(self, x):
        feat = self.conv(x)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureAlign(nn.Module):
    # def __init__(self, fsm, in_nc=128, out_nc=128):
    def __init__(self, fsm, out_nc=128):
        super(FeatureAlign, self).__init__()
        # self.fsm = FeatureSelectionModule(in_nc, out_nc)
        self.fsm = fsm
        self.offset = ConvBNReLU(out_nc * 2, out_nc, 1, 1, 0)
        self.dcpack_L2 = DCN(out_nc, out_nc // 2, 3, stride=1, padding=1, dilation=1,
                             deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        self.fsm_cat = ConvBNReLU(out_nc // 2, out_nc, 1, 1, 0)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='nearest')
        else:
            feat_up = feat_s
        feat_arm = self.fsm(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        # fcat = torch.cat([feat_arm, feat_align], dim=1)
        feat = self.fsm_cat(feat_align) + feat_arm

        return feat, feat_arm


class ContextPath(nn.Module):
    def __init__(self, backbone):
        super(ContextPath, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class Context(nn.Module):
    # def __init__(self, fsm, in_nc=128, out_nc=128):
    def __init__(self, fsm):
        super(Context, self).__init__()
        # self.fsm = FeatureSelectionModule(in_nc, out_nc)
        self.fsm = fsm
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, feat32):
        feat32 = self.fsm(feat32)  # 0~1 * feats
        return feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class GenFANet(nn.Module):
    """ generic FANet model """
    def __init__(self,
                 contextpath: ContextPath,
                 context: Context,
                 align16: FeatureAlign,
                 align8: FeatureAlign,
                 out32: BiSeNetOutput,
                 out16: BiSeNetOutput,
                 out8: BiSeNetOutput,
                 out_nc=128, n_classes=19, output_aux=False):
        super(GenFANet, self).__init__()

        self.out_nc = out_nc
        # self.nc_8, self.nc_16, self.nc_32 = feature_dim

        self.n_classes = n_classes
        self.output_aux = output_aux

        # self.cp = ContextPath(backbone)
        self.cp = contextpath

        # self.context = Context(in_nc=self.nc_32, out_nc=self.out_nc)
        # self.context = Context(FeatureSelectionModule(self.nc_32, self.out_nc))
        self.context = context
        # self.align16 = FeatureAlign(in_nc=self.nc_16, out_nc=self.out_nc)
        # self.align16 = FeatureAlign(FeatureSelectionModule(self.nc_16, self.out_nc), out_nc=self.out_nc)
        self.align16 = align16
        # self.align8 = FeatureAlign(in_nc=self.nc_8, out_nc=self.out_nc)
        # self.align8 = FeatureAlign(FeatureSelectionModule(self.nc_8, self.out_nc), out_nc=self.out_nc)
        self.align8 = align8

        # self.conv_out = BiSeNetOutput(self.out_nc, self.out_nc // 2, self.n_classes)
        self.conv_out = out8
        # self.conv_out16 = BiSeNetOutput(self.out_nc, self.out_nc // 2, self.n_classes)
        self.conv_out16 = out16
        # self.conv_out32 = BiSeNetOutput(self.out_nc, self.out_nc // 2, self.n_classes)
        self.conv_out32 = out32
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_res16, feat_res32 = self.cp(x)
        feat32_context = self.context(feat_res32)
        feat16_align, feat16_detail = self.align16(feat_res16, feat32_context)
        feat8_align, feat8_detail = self.align8(feat_res8, feat16_align)
        feat_out = self.conv_out(feat8_align)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        if self.output_aux:
            aux_out32 = F.interpolate(self.conv_out32(feat32_context), (H, W), mode='bilinear', align_corners=True)
            aux_out16 = F.interpolate(self.conv_out16(feat16_align), (H, W), mode='bilinear', align_corners=True)
            return feat_out, aux_out16, aux_out32
        else:
            return feat_out, None, None

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureAlign, Context, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

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
        super(GenFANet, self).load_state_dict(model_dict)


class FANetBasicBlock(GenFANet):
    def __init__(self,
                 # backbone related settings
                 d=None,  # R18d -> d=[1, 1, 0, 1]; R34d -> d=[2, 3, 4, 2]
                 e=None,  # [1.0] * 16 -> [64, 128, 256, 512]
                 w=None,  # [1.0] * 5
                 out_nc=128, n_classes=19, output_aux=False):

        supernet = GenOFABasicBlockNets(
            depth_list=[2, 3, 4, 2], expand_ratio_list=[0.65, 0.8, 1.0], width_mult_list=[0.65, 0.8, 1.0],
            features_only=True)
        supernet.set_active_subnet(d=d, e=e, w=w)
        backbone = supernet.get_active_subnet(preserve_weight=True)

        # self.out_nc = 128
        nc_8, nc_16, nc_32 = backbone.config['feature_dim']

        # self.n_classes = n_classes
        # self.output_aux = output_aux

        cp = ContextPath(backbone)

        # self.context = Context(in_nc=self.nc_32, out_nc=self.out_nc)
        context = Context(FeatureSelectionModule(ConvBNReLU(nc_32, out_nc, ks=1, stride=1, padding=0)))

        # self.align16 = FeatureAlign(in_nc=self.nc_16, out_nc=self.out_nc)
        align16 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_16, out_nc, ks=1, stride=1, padding=0)), out_nc=out_nc)

        # self.align8 = FeatureAlign(in_nc=self.nc_8, out_nc=self.out_nc)
        align8 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_8, out_nc, ks=1, stride=1, padding=0)), out_nc=out_nc)

        conv_out = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out16 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out32 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)

        super(FANetBasicBlock, self).__init__(cp, context, align16, align8, conv_out32, conv_out16, conv_out,
                                              out_nc, n_classes, output_aux)


class FANet(GenFANet):
    def __init__(self,
                 # backbone related settings
                 backbone='resnet101d',  # should be one the backbone from timm.models
                 feature_dim=(512, 1024, 2048),
                 # https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                 out_nc=128, n_classes=19, output_aux=False, pretrained_backbone=True):

        backbone = create_model(backbone, pretrained=pretrained_backbone, features_only=True, out_indices=(2, 3, 4))

        # self.out_nc = 128
        nc_8, nc_16, nc_32 = feature_dim

        # self.n_classes = n_classes
        # self.output_aux = output_aux

        cp = ContextPath(backbone)

        # self.context = Context(in_nc=self.nc_32, out_nc=self.out_nc)
        context = Context(FeatureSelectionModule(ConvBNReLU(nc_32, out_nc, ks=1, stride=1, padding=0)))

        # self.align16 = FeatureAlign(in_nc=self.nc_16, out_nc=self.out_nc)
        align16 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_16, out_nc, ks=1, stride=1, padding=0)), out_nc=out_nc)

        # self.align8 = FeatureAlign(in_nc=self.nc_8, out_nc=self.out_nc)
        align8 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_8, out_nc, ks=1, stride=1, padding=0)), out_nc=out_nc)

        conv_out = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out16 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out32 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)

        super(FANet, self).__init__(cp, context, align16, align8, conv_out32, conv_out16, conv_out,
                                        out_nc, n_classes, output_aux)


if __name__ == '__main__':

    # fanet = FANetR18d(d=[0, 0, 0, 0], e=[0.65] * 16, w=[0] * 5)
    # fanet = FANet(backbone='resnet18d', feature_dim=(128, 256, 512), pretrained_backbone=False)
    fanet = FANet(backbone='resnet152d', feature_dim=(512, 1024, 2048), pretrained_backbone=False)
    print(fanet)

    # weights = torch.load("pretrained/resnet12d.pth", map_location='cpu')
    # fanet.load_state_dict(weights)

    data = torch.rand(1, 3, 512, 1024)
    fanet.eval()
    out = fanet(data)
    print(out[0].size())