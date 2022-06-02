import random
import numpy as np

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3


class GenOFAMobileNetV3:
    def __init__(self, image_scale_list, width_mult_list, ks_list, expand_ratio_list, depth_list,
                 n_classes=1000, dropout_rate=0.):

        self.image_scale_list = image_scale_list
        self.width_mult_list = width_mult_list
        self.ks_list = ks_list
        self.expand_ratio_list = expand_ratio_list
        self.depth_list = depth_list
        self.active_width_mult_idx = 0

        self.engine = [
            OFAMobileNetV3(
                n_classes=n_classes, dropout_rate=dropout_rate, width_mult=wid_mult,
                ks_list=ks_list, expand_ratio_list=expand_ratio_list,
                depth_list=depth_list) for wid_mult in width_mult_list
        ]

    def forward(self, x):
        return self.engine[self.active_width_mult_idx](x)

    def parameters(self):
        return self.engine[self.active_width_mult_idx].parameters()

    def cuda(self):
        for i in range(len(self.engine)):
            self.engine[i] = self.engine[i].cuda()

    def sample_active_subnet(self):

        image_scale = random.choice(self.image_scale_list)
        wid_mult_idx = random.choice(range(len(self.width_mult_list)))
        sub_str = self.engine[wid_mult_idx].sample_active_subnet()

        self.active_width_mult_idx = wid_mult_idx

        return {'r': image_scale, 'w': self.width_mult_list[wid_mult_idx], **sub_str}

    def set_active_subnet(self, w=None, ks=None, e=None, d=None, **kwargs):

        if w is None:
            wid_mult_idx = self.active_width_mult_idx
        else:
            wid_mult_idx = np.where(w == np.array(self.width_mult_list))[0][0]
            self.active_width_mult_idx = wid_mult_idx

        self.engine[wid_mult_idx].set_active_subnet(ks=ks, e=e, d=d, **kwargs)

    def get_active_subnet(self, preserve_weight=True):
        return self.engine[self.active_width_mult_idx].get_active_subnet(preserve_weight)

    def load_state_dict(self, state_dict_list):
        for i, state_dict in enumerate(state_dict_list):
            self.engine[i].load_state_dict(state_dict)
