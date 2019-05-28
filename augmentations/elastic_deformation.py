import random
import numpy as np
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask
from scipy import ndimage
from elasticdeform import deform_random_grid


class ElasticDeformation(BaseAugmentation):
    def __init__(self, sigma_std, possible_points):
        self.sigma_std = sigma_std
        self.possible_points = possible_points
        super().__init__()

    def execute(self, image, mask):
        fn = random_elastic_deform_fn(
            self.sigma_std, self.possible_points)
        return fn(image, mask)


def elastic_deform(img, mask, sigma, points):
    [img, mask] = deform_random_grid(
        [img, mask], sigma=sigma, points=points,  axis=[(0, 1, 2), (0, 1, 2)])
    return img, round_mask(mask)


def random_elastic_deform_fn(sigma_std, possible_points):
    sigma = scale_truncated_norm(sigma_std)
    points = np.random.choice(possible_points)
    return lambda img, mask: elastic_deform(img, mask, sigma, points)
