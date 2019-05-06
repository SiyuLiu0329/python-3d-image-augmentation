import numpy as np
import scipy.ndimage as ndimage
import random
import skimage.transform as transform
from scipy.stats import truncnorm
from elasticdeform import deform_random_grid


AXES = [(0, 1), (1, 2), (0, 2)]


def to_channels(arr):
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),))
    for c in channels:
        c = int(c)
        res[:, :, :, c:c+1][arr == c] = 1

    return res


def combine_channels(arr):
    w, h, d, c = arr.shape
    res = np.zeros((w, h, d, 1))
    for i in range(c):
        res[:, :, :, 0][arr[:, :, :, i] == 1] = i
    return res


### Rotation ###
def rotate(img, deg, ax, is_mask):
    ax1, ax2 = ax
    if ax1 > 2 or ax2 > 2:
        raise NotImplementedError
    img = ndimage.rotate(img, deg, ax, reshape=False, prefilter=True)
    return round_mask(img) if is_mask else img


def random_rotation_fn(std):
    d = scale_truncated_norm(std)
    a = random.choice(AXES)
    return lambda img, is_mask: rotate(img, d, a, is_mask)


### shift ###
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


### elastic deform ###
def elastic_deform(img, mask, sigma, points):
    [img, mask] = deform_random_grid(
        [img, mask], sigma=sigma, points=points,  axis=[(0, 1, 2), (0, 1, 2)])
    return img, round_mask(mask)


def random_elastic_deform_fn(sigma_std, possible_points):
    sigma = scale_truncated_norm(sigma_std)
    points = np.random.choice(possible_points)
    return lambda img, mask: elastic_deform(img, mask, sigma, points)


# Swirl
def swirl(img, ax, strenght, radius, is_mask):
    ax1, ax2 = ax
    swapped = np.swapaxes(img, ax1, ax2)
    if is_mask:
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


def random_swirl_fn(strength_std, r):
    ax = random.choice(AXES)
    s = scale_truncated_norm(strength_std)
    return lambda img, is_mask: swirl(img, ax, s, r, is_mask)


### Utils ###
def round_mask(m):
    m[m > 0.5] = 1
    m[m <= 0.5] = 0
    return m


def scale_truncated_norm(std):
    res = truncnorm.rvs(-1, 1, loc=0, scale=std, size=1)
    return res
