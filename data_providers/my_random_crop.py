import torchvision.transforms as transforms


class MyRandomCrop(transforms.RandomCrop):
    ACTIVE_SIZE = 224
    IMAGE_SIZE_LIST = [224]
    IMAGE_SIZE_SEG = 4

    CONTINUOUS = False
    SYNC_DISTRIBUTED = True

    EPOCH = 0
    BATCH = 0