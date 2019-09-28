#import random
import numpy as np
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask, round_mask_semantic
#from scipy import ndimage
from elasticdeform import deform_random_grid


class ElasticDeformation(BaseAugmentation):
    def __init__(self, sigma_std, possible_points, categorical=True):
        self.sigma_std = sigma_std
        self.possible_points = possible_points
        self.categorical = categorical
        super().__init__()
        
    def random_elastic_deform_fn(self, sigma_std, possible_points):
        sigma = scale_truncated_norm(sigma_std)
        points = np.random.choice(possible_points)
        return lambda img, mask: self.elastic_deform(img, mask, sigma, points)
    
    def elastic_deform(self, img, mask, sigma, points):
        [img, mask] = deform_random_grid(
            [img, mask], sigma=sigma, points=points,  axis=[(0, 1, 2), (0, 1, 2)])
        
        if self.categorical:
            mask = round_mask(mask)
        else:
            mask = round_mask_semantic(mask)
        
        return img, mask

    def execute(self, image, mask):
        fn = self.random_elastic_deform_fn(
            self.sigma_std, self.possible_points)
        return fn(image, mask)
