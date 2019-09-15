import random
import numpy as np
import skimage.transform as transform
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask
#from scipy import ndimage

AXES = [(0, 1), (1, 2), (0, 2)]

class Swirl(BaseAugmentation):
    def __init__(self, strength_std, radius, categorical=True):
        self.strength_std = strength_std
        self.radius = radius
        self.categorical = categorical
        super().__init__()
        
    def random_swirl_fn(self, strength_std, r):
        ax = random.choice(AXES)
        s = scale_truncated_norm(strength_std)
        return lambda img, is_mask: self.swirl(img, ax, s, r, is_mask)
    
    def swirl(self, img, ax, strenght, radius, is_mask):
        ax1, ax2 = ax
        swapped = np.swapaxes(img, ax1, ax2)
        if is_mask and self.categorical:
            _, _, _, channels = img.shape
            swapped = [transform.swirl(swapped[:, :, :, c], rotation=0,
                                       strength=strenght, radius=radius)
                       for c in range(channels)]
            swapped = np.stack(swapped, axis=-1)
            swapped = np.swapaxes(swapped, ax1, ax2)
            return round_mask(swapped)
    
        swapped = transform.swirl(swapped[:, :, :, 0], rotation=0,
                                  strength=strenght, radius=radius)
        swapped = swapped.reshape(swapped.shape + (1,))
        swapped = np.swapaxes(swapped, ax1, ax2)
        
        return swapped

    def execute(self, image, mask):
        fn = self.random_swirl_fn(self.strength_std, self.radius)
        return fn(image, False), fn(mask, True)
