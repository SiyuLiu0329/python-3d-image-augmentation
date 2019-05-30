import numpy as np
import bezier
# import matplotlib.pyplot as plt
from augmentations.base_augmentation import BaseAugmentation
from utils import scale_truncated_norm
from scipy import ndimage


class BezierLUT(BaseAugmentation):
    def __init__(self, xs, ys, y_std, degree):
        self.xs = xs
        self.ys = ys
        self.y_std = y_std
        self.degree = degree
        super().__init__()

    def execute(self, image, mask):
        fn = random_bezier_lut_fn(
            self.xs, self.ys, self.y_std, self.degree)
        return fn(image, False), fn(mask, True)


class LUT:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.lut = {}
        self._step_size = 0.00001
        self._build_lut()

    def _build_lut(self):
        for i in range(len(self.xs)):
            x, y = self.xs[i], self.ys[i]
            x_r = round(x, 5)
            self.lut[x_r] = y

    def look_up(self, value):
        if value == 1 or value == 0:
            return value
        y = self._get_neighbour(value)

        return y

    def _get_neighbour(self, value):
        value = round(value, 5)
        delta_pos = 0
        delta_neg = 0
        y = None
        while True:
            y = self.lut.get(value + delta_pos)
            if y is not None:
                break
            y = self.lut.get(value + delta_neg)
            if y is not None:
                break

            delta_pos += self._step_size
            delta_neg -= self._step_size

        return y

    @classmethod
    def from_bezier_curve(clz, curve):
        res = curve.evaluate_multi(np.linspace(0.0, 1.0, 100000))
        return clz(res[0], res[1])


def apply_bezier_lut(img, curve, is_mask):
    if is_mask:
        return img

    # change the range to 0 - 1
    img = (img - img.min()) / (img.max() - img.min())

    lut = LUT.from_bezier_curve(curve)

    # x = np.linspace(0, 1, 100)
    # plt.plot(x, [lut.look_up(e) for e in x])
    # plt.show()

    def lut_fn(value):
        return lut.look_up(value)

    img = np.vectorize(lut_fn)(img)
    return img


def random_bezier_lut_fn(xs, ys, y_std, degree):
    def apply_variability(value, std):
        res = value + scale_truncated_norm(std)[0]
        if res > 1:
            res = 1
        if res < 0:
            res = 0
        return res

    ys = [apply_variability(value, y_std) for value in ys]
    xs = [0.0] + xs + [1.0]
    ys = [0.0] + ys + [1.0]
    # print(ys)
    nodes = np.asfortranarray([
        xs, ys
    ])
    curve = bezier.Curve(nodes, degree=degree)
    return lambda img, is_mask: apply_bezier_lut(img, curve, is_mask)
