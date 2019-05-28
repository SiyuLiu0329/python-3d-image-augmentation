import numpy as np
import bezier
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm, round_mask
from scipy import ndimage

import matplotlib.pyplot as plt


class BezierLUT(BaseAugmentation):
    def __init__(self, xs, ys, degree):
        self.xs = xs
        self.ys = ys
        self.degree = degree
        super().__init__()

    def execute(self, image, mask):
        fn = random_bezier_lut_fn(self.xs, self.ys, self.degree)
        return fn(image, False), fn(mask, True)


def apply_bezier_lut(img, curve, is_mask):
    if is_mask:
        return img

    # change the range to 0 - 1
    img = (img - img.min()) / (img.max() - img.min())

    def lut_fn(value):
        return curve.evaluate(value)[1]

    img = np.vectorize(lut_fn)(img)
    return img


def random_bezier_lut_fn(xs, ys, degree):
    xs = [0.0] + xs + [1.0]
    ys = [0.0] + ys + [1.0]
    nodes = np.asfortranarray([
        xs, ys
    ])
    curve = bezier.Curve(nodes, degree=degree)
    # debug
    x = np.linspace(0, 1, 50)
    plt.plot(x, curve.evaluate_multi(x)[1, :])
    plt.scatter(xs, ys)
    plt.show()

    return lambda img, is_mask: apply_bezier_lut(img, curve, is_mask)
