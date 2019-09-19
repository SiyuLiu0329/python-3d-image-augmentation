import random
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask, round_mask_semantic
from scipy import ndimage

AXES = [(0, 1), (1, 2), (0, 2)]


class Rotation(BaseAugmentation):
    def __init__(self, std, categorical=True):
        self.std = std
        self.categorical = categorical
        super().__init__()

    def random_uniaxial_rotation_fn(self, std):
        d = scale_truncated_norm(std)
        a = random.choice(AXES)
        return lambda img, is_mask: self.rotate(img, d, a, is_mask)

    def execute(self, image, mask):
        fn = self.random_uniaxial_rotation_fn(self.std)
        return fn(image, False), fn(mask, True)

    def rotate(self, img, deg, ax, is_mask):
        ax1, ax2 = ax
        if ax1 > 2 or ax2 > 2:
            raise NotImplementedError
        img = ndimage.rotate(img, deg[0], ax, reshape=False, prefilter=True)
        
        if is_mask:
            if self.categorical:
                return round_mask(img)
            else:
                return round_mask_semantic(img)
        
        return img
