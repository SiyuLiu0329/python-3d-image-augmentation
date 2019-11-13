#import random
import numpy as np
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm
from elasticdeform import deform_grid


class ElasticDeformation(BaseAugmentation):
    def __init__(self, sigma_std, possible_points, categorical=True):
        self.sigma_std = sigma_std
        self.possible_points = possible_points
        self.categorical = categorical
        super().__init__()

    def random_displacement_grid(self, X, sigma=25, points=3, axis=None):
        """
        This generates a random, square deformation grid with displacements
        sampled from from a normal distribution with standard deviation `sigma`.
        """
        # prepare inputs and axis selection
        axis, deform_shape = _normalize_axis_list(axis, [X])

        if not isinstance(points, (list, tuple)):
            points = [points] * len(deform_shape)

        displacement = np.random.randn(len(deform_shape), *points) * sigma
        return displacement

    def random_elastic_deform_fn(self, sigma_std, possible_points):
        sigma = scale_truncated_norm(sigma_std)
        points = np.random.choice(possible_points)
        return lambda img, mask: self.elastic_deform(img, mask, sigma, points)

    def elastic_deform(self, img, mask, sigma, points):
        disp_grid = self.random_displacement_grid(
            img, sigma, points, axis=(0, 1, 2))
#        [img, mask] = deform_random_grid(
#            [img, mask], sigma=sigma, points=points, axis=[(0, 1, 2), (0, 1, 2)], order=order)
        img = deform_grid(img, disp_grid, axis=(0, 1, 2), order=3)
        mask = deform_grid(mask, disp_grid, axis=(0, 1, 2), order=0)

#        if self.categorical:
#            mask = round_mask(mask)
#        else:
#            mask = round_mask_semantic(mask)

        return img, mask

    def execute(self, image, mask):
        fn = self.random_elastic_deform_fn(
            self.sigma_std, self.possible_points)
        return fn(image, mask)


def _normalize_axis_list(axis, Xs):
    if axis is None:
        axis = [tuple(range(x.ndim)) for x in Xs]
    elif isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = [axis] * len(Xs)
    assert len(axis) == len(
        Xs), 'Number of axis tuples should match number of inputs.'
    input_shapes = []
    for x, ax in zip(Xs, axis):
        assert isinstance(ax, tuple), 'axis should be given as a tuple'
        assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
        assert len(ax) == len(
            axis[0]), 'All axis tuples should have the same length.'
        assert ax == tuple(set(ax)), 'axis must be sorted and unique'
        assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
        input_shapes.append(tuple(x.shape[d] for d in ax))
    assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
    deform_shape = input_shapes[0]
    return axis, deform_shape
