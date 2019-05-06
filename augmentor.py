import random
import inspect
import inspect
import matplotlib.pyplot as plt
from augmentation_utils import random_affine_warp_fn, random_elastic_deform_fn, random_rotation_fn, random_shift_fn, random_swirl_fn, combine_channels


class Augmentor:
    def __init__(self):
        self._augmentation_fns = []

    def apply_augmentation_to_batch(self, x_batch, y_batch, debug=False):
        # x: (w, h, d, 1)
        # y: (w, h, d, c)
        if len(self._augmentation_fns) == 0:
            raise ValueError(
                "The augmentor does not have any augmentation function - add augmentation functions before applying augmentation.")
        assert len(x_batch) == len(y_batch)
        for i in range(len(x_batch)):
            x, y = x_batch[i], y_batch[i]
            x, y = self._do_aug(x, y, debug=debug)

            x_batch[i] = x
            y_batch[i] = y

        return x_batch, y_batch,

    def add_rotation_fn(self, std):
        self._augmentation_fns.append(lambda: random_rotation_fn(std))

    def add_shift_fn(self, shift_stds):
        self._augmentation_fns.append(lambda: random_shift_fn(shift_stds))

    def add_swirl_fn(self, strength_std, radius):
        self._augmentation_fns.append(
            lambda: random_swirl_fn(strength_std, radius))

    def add_elastic_deformation_fn(self, sigma_std, possible_points):
        self._augmentation_fns.append(
            lambda: random_elastic_deform_fn(sigma_std, possible_points))

    def add_affine_warp_fn(self, vertex_percentage_std):
        self._augmentation_fns.append(
            lambda: random_affine_warp_fn(vertex_percentage_std))

    def _do_aug(self, x, y, debug=False):
        fn = random.choice(self._augmentation_fns)()
        fn_args = inspect.getargspec(fn).args
        if "img" in fn_args and "mask" in fn_args:
            x, y = fn(x, y)
        else:
            x, y = fn(x, False), fn(y, True)
        if debug:
            self._debug_show(x, y, fn)
        return x, y

    def _debug_show(self, x, y, fn):
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
