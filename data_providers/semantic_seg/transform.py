#!/usr/bin/python
# -*- encoding: utf-8 -*-

import random
import numpy as np
from PIL import Image
from enum import Enum
import PIL.ImageEnhance as ImageEnhance
from collections.abc import Sequence, Iterable

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


def larger_edge_resize(img: Image, size, interpolation=Image.BICUBIC):
    """ Resize the given image to the target size. If size is a single number, the larger edge will be resized
    to that scale maintaining the image's aspect ratio.

    Args:
        img (PIL.Image): Input image.
        size (int or list of int): Target size
        interpolation (int): Interpolation type: Image.NEAREST, Image.BILINEAR, or Image.BICUBIC.

    Returns:

    """
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w >= h and w == size) or (h >= w and h == size):
            return img
        if w < h:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
        else:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


# Borrowed from: https://github.com/pytorch/vision/blob/v0.9.1/torchvision/transforms/functional.py
class InterpolationMode(Enum):
    """Interpolation modes
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


# Borrowed from: https://github.com/pytorch/vision/blob/v0.9.1/torchvision/transforms/functional.py
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


class SegTransform(object):
    pass


class LargerEdgeResize(SegTransform, transforms.Resize):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            larger edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * width / height)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BICUBIC``
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        super(LargerEdgeResize, self).__init__(size, interpolation)

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = larger_edge_resize(img, self.size, self.interpolation)
        lbl = larger_edge_resize(lbl, self.size, Image.NEAREST)

        return img, lbl

    def __repr__(self):
        interpolate_str = _interpolation_modes_from_int[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def call_recursive(f, x):
    return [call_recursive(f, y) for y in x] if isinstance(x, (list, tuple)) else f(x)


class ToTensor(SegTransform):
    """ Convert an image and pose in numpy.ndarray format to Tensor.

    Convert a numpy.ndarray image (H x W x C) in the range [0, 255] and numpy.ndarray pose (3)
    to torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] and torch.FloatTensor of shape (3)
    correspondingly.
    """

    def __call__(self, img, lbl):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        return call_recursive(F.to_tensor, img), torch.from_numpy(np.array(lbl).astype('long'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False):
        super(Normalize, self).__init__(mean, std, inplace)

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(x, self.mean, self.std, self.inplace)


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> img_landmarks_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3) or bounding box (4)
        Returns:
            Tensor or list of Tensor: Transformed images or poses
        """
        assert len(args) == 2 or (isinstance(args[0], (list, tuple)) and len(args[0]) == 2), \
            'Two arguments must be specified, an image and a corresponding label'
        input = list(args) if len(args) > 1 else args[0]
        for t in self.transforms:
            if isinstance(t, SegTransform):
                input = list(t(*input))
            else:
                input[0] = call_recursive(t, input[0])

        return tuple(input)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomResize(SegTransform):
    def __init__(self, p=0.5, scale_range=None, scale_values=None, interpolation=Image.BICUBIC):
        assert (scale_range is None) ^ (scale_values is None)
        self.p = p
        self.scale_range = scale_range
        self.scale_values = scale_values
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        if random.random() >= self.p:
            return img, lbl
        if self.scale_range is not None:
            scale = random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        else:
            scale = self.scale_values[random.randint(0, len(self.scale_values))]

        size = tuple(np.round(np.array(img.size[::-1]) * scale).astype(int))
        img = F.resize(img, size, self.interpolation)
        lbl = F.resize(lbl, size, Image.NEAREST)

        return img, lbl


class RandomCrop(SegTransform, transforms.RandomCrop):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, lbl_fill=None, padding_mode='constant'):
        super(RandomCrop, self).__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.lbl_fill = fill if lbl_fill is None else lbl_fill

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.

        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            lbl = F.pad(lbl, self.padding, self.lbl_fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            lbl = F.pad(lbl, (self.size[1] - lbl.size[0], 0), self.lbl_fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            lbl = F.pad(lbl, (0, self.size[0] - lbl.size[1]), self.lbl_fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)


class RandomHorizontalFlip(SegTransform):
    """Horizontally flip the given image and its corresponding label randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = F.hflip(img)
            lbl = F.hflip(lbl)

        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
