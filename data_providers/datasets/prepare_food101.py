# Steps to prepare food-101 dataset
# Download Food-101 dataset from "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
# untar the dataset "tar -xvzf food-101.tar.gz"
# put images to respective folders as described in "/food-101/meta/train.txt" and "/food-101/meta/test.txt"

import os
from shutil import copyfile

_data_root = '/home/cseadmin/datasets/food-101'  # update to your own path
_images = os.path.join(_data_root, 'images')
_classes = os.path.join(_data_root, 'meta', 'classes.txt')
_train_split = os.path.join(_data_root, 'meta', 'train.txt')
_test_split = os.path.join(_data_root, 'meta', 'test.txt')

# make a directory for train and test set
os.makedirs(os.path.join(_data_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(_data_root, 'test'), exist_ok=True)

# read classes, train, and test files
with open(_classes, 'r') as fh:
    for line in fh:
        os.makedirs(os.path.join(_data_root, 'train', line.rstrip().strip('\n')), exist_ok=True)
        os.makedirs(os.path.join(_data_root, 'test', line.rstrip().strip('\n')), exist_ok=True)

train_image_list = []
with open(_train_split, 'r') as fh:
    for line in fh:
        train_image_list.append(line.rstrip().strip('\n') + '.jpg')

test_image_list = []
with open(_test_split, 'r') as fh:
    for line in fh:
        test_image_list.append(line.rstrip().strip('\n') + '.jpg')

# place train images into respective folders
for img in train_image_list:
    copyfile(os.path.join(_images, img), os.path.join(_data_root, 'train', img))

# place test images into respective folders
for img in test_image_list:
    copyfile(os.path.join(_images, img), os.path.join(_data_root, 'test', img))