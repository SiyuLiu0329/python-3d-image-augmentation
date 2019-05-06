import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import random
import inspect
import skimage.transform as transform
from scipy.stats import truncnorm
from elasticdeform import deform_random_grid

# *_std: std for truncated norm
CONFIG = {
    "rotation_std": 20,
    "shift_stds": [20, 20, 20],
    "elastic_sigma_std": 3,
    "elastic_points": [3, 4, 5],
    "swirl_strength_std": 1
}


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


def random_rotation_fn():
    d = scale_truncated_norm(CONFIG["rotation_std"])
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


def random_shift_fn():
    shifts = CONFIG["shift_stds"]
    shifts = [scale_truncated_norm(s) for s in shifts]
    return lambda img, is_mask: shift(img, shifts, is_mask)


### elastic deform ###
def elastic_deform(img, mask, sigma, points):
    [img, mask] = deform_random_grid(
        [img, mask], sigma=sigma, points=points,  axis=[(0, 1, 2), (0, 1, 2)])
    return img, round_mask(mask)


def random_elastic_deform_fn():
    sigma = scale_truncated_norm(CONFIG["elastic_sigma_std"])
    points = np.random.choice(CONFIG["elastic_points"])
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


def random_swirl_fn():
    ax = random.choice(AXES)
    r = 300
    s = CONFIG["swirl_strength_std"]
    s = scale_truncated_norm(s)
    return lambda img, is_mask: swirl(img, ax, s, r, is_mask)


### Utils ###
def round_mask(m):
    m[m > 0.5] = 1
    m[m <= 0.5] = 0
    return m


def scale_truncated_norm(std):
    res = truncnorm.rvs(-1, 1, loc=0, scale=std, size=1)
    return res


AXES = [(0, 1), (1, 2), (0, 2)]
MODES = [random_elastic_deform_fn, random_swirl_fn,
         random_shift_fn, random_rotation_fn]


def do_aug(x, y, debug=False):
    fn = random.choice(MODES)()
    fn_args = inspect.getargspec(fn).args
    if "img" in fn_args and "mask" in fn_args:
        x, y = fn(x, y)
    else:
        x, y = fn(x, False), fn(y, True)
    if debug:
        debug_show(x, y, fn)
    return x, y


def apply_augmentation(x_batch, y_batch, debug=False):
    # x_batch: (?, w, h, d, 1)
    # y_batch: (?, w, h, d, c)
    assert len(x_batch) == len(y_batch)
    for i in range(len(x_batch)):
        x, y = x_batch[i], y_batch[i]
        x, y = do_aug(x, y, debug=debug)

        x_batch[i] = x
        y_batch[i] = y

    return x_batch, y_batch,


def debug_show(x, y, fn):
    # x: (w, h, d, 1)
    # y: (w, h, d, c)
    y = combine_channels(y)
    w, h, d, _ = x.shape
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(x[w//2, :, :, 0], cmap='gray')
    plt.imshow(y[w//2, :, :, 0], alpha=0.4)

    ax = plt.subplot(1, 3, 2)
    ax.set_title(str(fn))
    plt.imshow(x[:, :, d//2, 0], cmap='gray')
    plt.imshow(y[:, :, d//2, 0], alpha=0.4)

    plt.subplot(1, 3, 3)
    plt.imshow(x[:, h//2, :, 0], cmap='gray')
    plt.imshow(y[:, h//2, :, 0], alpha=0.4)

    plt.show()
    plt.close()
