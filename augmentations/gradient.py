import random
import numpy as np
import itertools
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask_semantic
from scipy import ndimage


class LinearGradient(BaseAugmentation):
    def __init__(self, gradient_stds):
        self.gradient_stds = gradient_stds
        super().__init__()

    def execute(self, image, mask):
        fn = random_linear_gradient_fn(self.gradient_stds)
        return fn(image, False), fn(mask, True)


def random_linear_gradient_fn(stds):
    gx, gy, gz = [scale_truncated_norm(s)[0] for s in stds]
    return lambda img, is_mask: apply_gradient(img, lambda x, y, z: gx * x + gy * y + gz * z, is_mask)


def apply_gradient(img, gradient_fn, is_mask):
    return img
