# Steps to prepare Flowers102 dataset
# Download Flowers102 dataset from "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
# Download Flowers102 dataset train/valid/test split from "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

import os
import glob
import scipy.io
import numpy as np
from shutil import copyfile, rmtree

_data_root = '/Users/luzhicha/Downloads/flowers102'  # update to your own path
_images = os.path.join(_data_root, 'jpg')
_meta = os.path.join(_data_root, 'setid.mat')
_label = os.path.join(_data_root, 'imagelabels.mat')

# make a directory for train and test set
os.makedirs(os.path.join(_data_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(_data_root, 'test'), exist_ok=True)

# read the data split mat file
meta_data = scipy.io.loadmat(_meta)
labels = scipy.io.loadmat(_label)['labels'][0]

for _class in np.unique(labels):
    os.makedirs(os.path.join(_data_root, 'train', str(_class)), exist_ok=True)
    os.makedirs(os.path.join(_data_root, 'test', str(_class)), exist_ok=True)
    os.makedirs(os.path.join(_data_root, '_tmp', str(_class)), exist_ok=True)  # for temporary holding images

# places these images into their classes folders
for i, label in enumerate(labels):
    copyfile(os.path.join(_images, 'image_{}.jpg'.format(str(i + 1).zfill(5))),
             os.path.join(_data_root, '_tmp', str(label), 'image_{}.jpg'.format(str(i + 1).zfill(5))))

# we combine trn and val to form our training set
trnid = np.concatenate((meta_data['trnid'][0], meta_data['valid'][0]))
tstid = meta_data['tstid'][0]

for _class in np.unique(labels):
    for img in glob.glob(os.path.join(_data_root, '_tmp', str(_class), "*.jpg")):
        img_name = os.path.basename(img)
        img_id = int(img_name.replace('.jpg', '').split('_')[1])
        if img_id in trnid:
            copyfile(img, os.path.join(_data_root, 'train', str(_class), img_name))
        elif img_id in tstid:
            copyfile(img, os.path.join(_data_root, 'test', str(_class), img_name))
        else:
            raise ValueError("img_id is neither in train or test split")

# remove the temporary placeholder
rmtree(os.path.join(_data_root, '_tmp'))

