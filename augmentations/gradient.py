import random
import numpy as np
import itertools
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask
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
    if is_mask:
        # no need to apply augmentation to masks
        return img

    d1, d2, _, _ = img.shape
    min_value = np.min(img)
    max_value = np.max(img)

    def to_coord(index):
        y, z = index // d1, index % d1
        x = y // d2
        y = y % d2
        return x, y, z

    def pixel_fn(index, value):
        x, y, z = to_coord(index)
        new_value = value + gradient_fn(x, y, z)
        if new_value > max_value:
            return max_value
        elif new_value < min_value:
            return min_value
        else:
            return new_value

    i = itertools.count(-1)
    img = np.vectorize(lambda value: pixel_fn(next(i), value))(img)

    return img
