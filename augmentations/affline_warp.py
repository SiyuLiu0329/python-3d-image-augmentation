#import random
import numpy as np
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask, round_mask_semantic
from scipy import ndimage


class AffineWarp(BaseAugmentation):
    def __init__(self, vertex_percentage_std, categorical=True):
        self.vertex_percentage_std = vertex_percentage_std
        self.categorical = categorical
        super().__init__()
        
    def random_affine_warp_fn(self, vertex_percentage_std):
        source_points = [[0, 0, 0], [0, 100, 0], [0, 0, 100], [100, 0, 0]]
        dest_points = np.array([
            [v + scale_truncated_norm(vertex_percentage_std)[0] for v in p] for p in source_points])
        return lambda img, is_mask: self.affine_warp(img, source_points, dest_points, is_mask)

    def execute(self, image, mask):
        fn = self.random_affine_warp_fn(self.vertex_percentage_std)
        return fn(image, False), fn(mask, True)

    def affine_warp(self, img, src, dest, is_mask):
        mat, _, _, _ = np.linalg.lstsq(src, dest, rcond=None)
        mat = np.array([np.append(row, 0) for row in mat])
        mat = np.concatenate([mat, [[0, 0, 0, 1]]], axis=0)
        if is_mask and self.categorical:
            _, _, _, channels = img.shape
            res = [ndimage.affine_transform(img[:, :, :, c], mat)
                   for c in range(channels)]
            res = np.stack(res, axis=-1)
            return round_mask(res)
        
        img = ndimage.affine_transform(img[:, :, :, 0], mat)
        
        if is_mask and not self.categorical:
            img = round_mask_semantic(img)
        
        return img.reshape(img.shape + (1,))
