#import random
import numpy as np
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask, round_mask_semantic
from scipy import ndimage


class Shifts(BaseAugmentation):
    def __init__(self, shifts_stds, categorical=True):
        self.shifts_stds = shifts_stds
        self.categorical = categorical
        super().__init__()
        
    def random_shift_fn(self, shift_stds):
        shifts = [scale_truncated_norm(s) for s in shift_stds]
        return lambda img, is_mask: self.shift(img, shifts, is_mask)

    def execute(self, image, mask):
        fn = self.random_shift_fn(self.shifts_stds)
        return fn(image, False), fn(mask, True)
    
    def shift(self, img, shifts, is_mask):
        order = 3 #bspline interp
        if is_mask:
            order = 0 #NN interp
            
            if self.categorical:
                _, _, _, channels = img.shape
                res = [ndimage.shift(img[:, :, :, c], shifts, order=order)
                       for c in range(channels)]
                res = np.stack(res, axis=-1)
#                return round_mask(res)
                return res #no rounding req as NN interp
    
        img = ndimage.shift(img[:, :, :, 0], shifts, order=order)
        
#        if is_mask and not self.categorical:
#            img = round_mask_semantic(img)
            
        return img.reshape(img.shape + (1,))
