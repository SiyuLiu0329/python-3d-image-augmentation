import numpy as np
from scipy.stats import truncnorm


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


def round_mask(m):
    m[m > 0.5] = 1
    m[m <= 0.5] = 0
    return m


def scale_truncated_norm(std):
    res = truncnorm.rvs(-1, 1, loc=0, scale=std, size=1)
    return res
