import random
import numpy as np
import cv2
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask_semantic, scale3d, to_channels
from scipy import ndimage


class Rescale(BaseAugmentation):
    def __init__(self, std):
        self.std = std
        super().__init__()

    def execute(self, image, mask):
        x, y, z, _ = image.shape
        image = image.reshape((x, y, z))
        mask = np.argmax(mask, axis=-1)
        newx = x + int(scale_truncated_norm(self.std))
        newy = y + int(scale_truncated_norm(self.std))
        newz = z + int(scale_truncated_norm(self.std))
        image = scale3d(image, newx, newy, newz, cv2.INTER_CUBIC)
        mask = scale3d(mask, newx, newy, newz, cv2.INTER_NEAREST)
        image = image.reshape(image.shape + (1,))
        mask = to_channels(mask)
        return image, mask


class RandomCrop(BaseAugmentation):
    def __init__(self, std):
        self.std = std
        super().__init__()

    def execute(self, image, mask):
        x, y, z, _ = image.shape
        dx, dy, dz = abs(int(scale_truncated_norm(self.std))),  abs(
            int(scale_truncated_norm(self.std))),  abs(int(scale_truncated_norm(self.std)))
        newx = x - dx
        newy = y - dy
        newz = z - dz
        sx, sy, sz = np.random.choice(range(dx)), np.random.choice(
            range(dy)), np.random.choice(range(dz))

        image = image[sx: sx + newx, sy: sy + newy, sz: sz + newz, :]
        mask = mask[sx: sx + newx, sy: sy + newy, sz: sz + newz, :]

        return image, mask
