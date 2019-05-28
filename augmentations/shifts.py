import random
import numpy as np
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask
from scipy import ndimage


class Shifts(BaseAugmentation):
    def __init__(self, shifts_stds):
        self.shifts_stds = shifts_stds
        super().__init__()

    def execute(self, image, mask):
        fn = random_shift_fn(self.shifts_stds)
        return fn(image, False), fn(mask, True)


def shift(img, shifts, is_mask):
    if is_mask:
        _, _, _, channels = img.shape
        res = [ndimage.shift(img[:, :, :, c], shifts)
               for c in range(channels)]
        res = np.stack(res, axis=-1)
        return round_mask(res)

    img = ndimage.shift(img[:, :, :, 0], shifts)
    return img.reshape(img.shape + (1,))


def random_shift_fn(shift_stds):
    shifts = [scale_truncated_norm(s) for s in shift_stds]
    return lambda img, is_mask: shift(img, shifts, is_mask)
