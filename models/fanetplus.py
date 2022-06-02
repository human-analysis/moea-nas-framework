import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from DCNv2.dcn_v2 import DCN
from timm import create_model

from inplace_abn import InPlaceABNSync as BatchNorm2d  # pip install inplace-abn
# from torch.nn import BatchNorm2d

from supernets.ofa_basic import GenOFABasicBlockNets


__all__ = ['GenFANetPlus', 'FeatureAlign', 'ContextPath', 'SelfAttn', 'Context',
           'BiSeNetOutput', 'FeatureSelectionModule', 'ConvBNReLU', 'Upsample']


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_chan, activation='identity')  # for inplace-bn
        # self.bn = BatchNorm2d(out_chan)
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


class SelfAttn(nn.Module):
    def __init__(self,
                 # dim_in,
                 dim_out,
                 conv1: ConvBNReLU, conv2: ConvBNReLU,
                 qkv_bias=False, qk_scale=None):
        super().__init__()
        # self.conv1 = ConvBNReLU(in_chan=dim_in, out_chan=dim_out, ks=1, padding=0)
        self.conv1 = conv1
        self.pos_embed = nn.Parameter(torch.zeros(1, 16*32, dim_out))
        self.norm1 = nn.LayerNorm(dim_out)
        head_dim = 32
        self.embed_size = [16, 32]
        num_heads = dim_out // head_dim
        self.q = nn.Linear(dim_out, dim_out, bias=qkv_bias)
        self.kv = nn.Linear(dim_out, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.conv2 = ConvBNReLU(in_chan=dim_out, out_chan=dim_out, ks=1, padding=0)
        self.conv2 = conv2
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def get_pos_embed(self, H, W):
        return F.interpolate(
            self.pos_embed.reshape(1, self.embed_size[0], self.embed_size[1], -1).permute(0, 3, 1, 2).contiguous(),
            size=(H, W), mode="bilinear", align_corners=True).reshape(1, -1, H * W).permute(0, 2, 1).contiguous()

    def forward_attn(self, x):
        q = self.q(x)
        kv = self.kv(x)
        q = rearrange(q, 'b nq (h d) -> b h nq d', h=self.num_heads)
        kv = rearrange(kv, 'b nk (h d) -> b h nk d', h=self.num_heads)
        k, v = kv, kv
        attn = (q @ rearrange(k, 'b h nk d -> b h d nk')) * self.scale
        attn = attn.softmax(dim=-1)
        x = rearrange(attn @ v, 'b h nq d -> b nq (h d)')
        return x

    def forward(self, x):
        x = self.conv1(x)
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2).contiguous()
        pos_embed = self.get_pos_embed(H, W)
        x = x + pos_embed
        x = x + self.forward_attn(self.norm1(x))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x + self.conv2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, factor=2, out_nc=128):
        super(Upsample, self).__init__()
        self.factor = factor
        self.proj = nn.Conv2d(out_nc, out_nc * factor * factor, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def forward(self, feat, HW):
        feat = self.up(self.proj(feat))
        feat = F.interpolate(feat, HW, mode='nearest')
        return feat


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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureAlign(nn.Module):
    # def __init__(self, fsm, in_nc=128, out_nc=128):
    def __init__(self, fsm, upsample, out_nc=128):
        super(FeatureAlign, self).__init__()
        # self.fsm = FeatureSelectionModule(in_nc, out_nc)
        # lateral connection
        self.fsm = fsm
        # feature upsample
        self.upsample = upsample
        # feature alignment
        self.offset = ConvBNReLU(out_nc * 2, out_nc, 1, 1, 0)
        self.dcpack_L2 = DCN(out_nc, out_nc, 3, stride=1, padding=1, dilation=1,
                             deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        # feature fusion
        self.fusion = ConvBNReLU(out_nc*2, out_nc, ks=1, stride=1, padding=0)
        # self.fsm_cat = ConvBNReLU(out_nc // 2, out_nc, 1, 1, 0)
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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        # selected features
        feat_arm = self.fsm(feat_l)  # 0~1 * feats
        # upsampled features
        feat_up = self.upsample(feat_s, HW)
        # aligned features
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        # fused features
        feat = self.fusion(torch.cat([feat_arm, feat_align], dim=1))
        return feat


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


class Context(nn.Module):  #
    # def __init__(self, fsm, in_nc=128, out_nc=128):
    def __init__(self, fsm, attn):
        super(Context, self).__init__()
        # self.fsm = FeatureSelectionModule(in_nc, out_nc)
        self.fsm = fsm
        # context modelling
        self.attn = attn
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
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
            elif isinstance(module, nn.LayerNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def forward(self, feat32):
        feat32_ = self.fsm(feat32)  # 0~1 * feats
        context = self.attn(feat32)
        return feat32_ + context


class GenFANetPlus(nn.Module):
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
        super(GenFANetPlus, self).__init__()

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
        feat16_align = self.align16(feat_res16, feat32_context)
        feat8_align = self.align8(feat_res8, feat16_align)
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
        super(GenFANetPlus, self).load_state_dict(model_dict)


class FANetBasicBlock(GenFANetPlus):
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
        context = Context(FeatureSelectionModule(
            ConvBNReLU(nc_32, out_nc, ks=1, stride=1, padding=0)), SelfAttn(nc_32, out_nc))

        # self.align16 = FeatureAlign(in_nc=self.nc_16, out_nc=self.out_nc)
        align16 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_16, out_nc, ks=1, stride=1, padding=0)), Upsample(factor=2, out_nc=out_nc), out_nc=out_nc)

        # self.align8 = FeatureAlign(in_nc=self.nc_8, out_nc=self.out_nc)
        align8 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_8, out_nc, ks=1, stride=1, padding=0)), Upsample(factor=2, out_nc=out_nc), out_nc=out_nc)

        conv_out = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out16 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out32 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)

        super(FANetBasicBlock, self).__init__(cp, context, align16, align8, conv_out32, conv_out16, conv_out,
                                              out_nc, n_classes, output_aux)


class FANetPlus(GenFANetPlus):
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
        context = Context(FeatureSelectionModule(
            ConvBNReLU(nc_32, out_nc, ks=1, stride=1, padding=0)),
            SelfAttn(out_nc, ConvBNReLU(in_chan=nc_32, out_chan=out_nc, ks=1, padding=0),
                     ConvBNReLU(in_chan=out_nc, out_chan=out_nc, ks=1, padding=0)))

        # self.align16 = FeatureAlign(in_nc=self.nc_16, out_nc=self.out_nc)
        align16 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_16, out_nc, ks=1, stride=1, padding=0)), Upsample(factor=2, out_nc=out_nc), out_nc=out_nc)

        # self.align8 = FeatureAlign(in_nc=self.nc_8, out_nc=self.out_nc)
        align8 = FeatureAlign(FeatureSelectionModule(
            ConvBNReLU(nc_8, out_nc, ks=1, stride=1, padding=0)), Upsample(factor=2, out_nc=out_nc), out_nc=out_nc)

        conv_out = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out16 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)
        conv_out32 = BiSeNetOutput(out_nc, out_nc // 2, n_classes)

        super(FANetPlus, self).__init__(cp, context, align16, align8, conv_out32, conv_out16, conv_out,
                                        out_nc, n_classes, output_aux)


if __name__ == '__main__':

    # fanet = FANetBasicBlock(d=[0, 0, 0, 0], e=[0.65] * 16, w=[0] * 5)
    fanetplus = FANetPlus(backbone='resnet18d', feature_dim=(128, 256, 512), pretrained_backbone=False)
    print(fanetplus)

    # weights = torch.load("pretrained/resnet12d.pth", map_location='cpu')
    # fanet.load_state_dict(weights)

    data = torch.rand(1, 3, 512, 1024)
    fanetplus.eval()
    out = fanetplus(data)
    print(out[0].size())