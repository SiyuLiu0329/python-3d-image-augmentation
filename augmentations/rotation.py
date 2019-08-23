import random
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask
from scipy import ndimage

AXES = [(0, 1), (1, 2), (0, 2)]


class Rotation(BaseAugmentation):
    def __init__(self, std):
        self.std = std
        super().__init__()

    def execute(self, image, mask):
        fn = random_uniaxial_rotation_fn(self.std)
        return fn(image, False), fn(mask, True)


def rotate(img, deg, ax, is_mask):
    ax1, ax2 = ax
    if ax1 > 2 or ax2 > 2:
        raise NotImplementedError
    img = ndimage.rotate(img, deg[0], ax, reshape=False, prefilter=True)
    return round_mask(img) if is_mask else img


def random_uniaxial_rotation_fn(std):
    d = scale_truncated_norm(std)
    a = random.choice(AXES)
    return lambda img, is_mask: rotate(img, d, a, is_mask)
