import numpy as np
import cv2
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


def normalize(img):
    return (img - img.mean()) / img.std()


def scale_truncated_norm(std, size=1):
    return truncnorm.rvs(-1, 1, loc=0, scale=std, size=size)


def round_mask_semantic(m):
    return np.around(m)


def scale3d(i, tar_x, tar_y, tar_z, interpolation):
    x, y, z = i.shape
    if x == tar_x and y == tar_y and z == tar_z:
        return i
    tar_arr = np.zeros((x, tar_y, tar_z))

    for axis1 in range(x):
        slice_2d = i[axis1, :, :]
        res = cv2.resize(slice_2d, (tar_z, tar_y), interpolation=interpolation)
        tar_arr[axis1, :, :] = res

    new_arr = np.zeros((tar_x, tar_y, tar_z))
    for axis3 in range(tar_z):
        slice_2d = tar_arr[:, :, axis3]
        res = cv2.resize(slice_2d, (tar_y, tar_x), interpolation=interpolation)
        new_arr[:, :, axis3] = res

    return new_arr
