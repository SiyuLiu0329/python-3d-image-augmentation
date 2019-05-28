import random
import inspect
import inspect
import matplotlib.pyplot as plt
import augmentation_utils as aug_utils


class Augmentor:
    def __init__(self):
        self._augmentation_fns = []
        self._sequencial_cacahe = None

    def add_sequence(self):
        assert self._sequencial_cacahe is None
        self._sequencial_cacahe = []
        return self

    def end_sequence(self):
        sequence = self._sequencial_cacahe
        assert sequence is not None
        assert len(sequence) != 0
        self._augmentation_fns.append(
            lambda: self.execute_sequence_fn(sequence))
        self._sequencial_cacahe = None
        return self

    def execute_sequence_fn(self, sequence):
        def execute_sequence(img, mask):
            for fn in sequence:
                img, mask = self._do_aug(img, mask, debug=False, fn=fn())
            return img, mask
        return execute_sequence

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

    def _fn_handler(self, fn):
        if self._sequencial_cacahe is not None:
            self._sequencial_cacahe.append(fn)
        else:
            self._augmentation_fns.append(fn)
        return self

    def add_uniaxial_rotation_fn(self, std):
        return self._fn_handler(lambda: aug_utils.random_uniaxial_rotation_fn(std))

    def add_shift_fn(self, shift_stds):
        return self._fn_handler(lambda: aug_utils.random_shift_fn(shift_stds))

    def add_swirl_fn(self, strength_std, radius):
        return self._fn_handler(lambda: aug_utils.random_swirl_fn(strength_std, radius))

    def add_elastic_deformation_fn(self, sigma_std, possible_points):
        return self._fn_handler(
            lambda: aug_utils.random_elastic_deform_fn(sigma_std, possible_points))

    def add_affine_warp_fn(self, vertex_percentage_std):
        return self._fn_handler(
            lambda: aug_utils.random_affine_warp_fn(vertex_percentage_std))

    def add_linear_gradient_fn(self, gradient_stds):
        assert len(gradient_stds) == 3
        return self._fn_handler(
            lambda: aug_utils.random_linear_gradient_fn(gradient_stds))

    def add_random_bezier_lut_fn(self, xs, ys, degree=2):
        return self._fn_handler(lambda: aug_utils.random_bezier_lut_fn(xs, ys, degree))

    def _do_aug(self, x, y, debug=False, fn=None):
        # calling this function will set the augmentation parameters
        fn = random.choice(self._augmentation_fns)() if fn is None else fn

        # some augmentation functions require the img and the mask to be process at the same time
        fn_args = inspect.getargspec(fn).args

        # apply augmentation by calling
        if "img" in fn_args and "mask" in fn_args:
            x, y = fn(x, y)
        else:
            x, y = fn(x, False), fn(y, True)
        if debug:
            self._debug_show(x, y, fn)
        return x, y

    def summary(self):
        print("=" * 15 + " Augmentation Functions " + "=" * 15)
        [print(fn) for fn in self._augmentation_fns]

    def _debug_show(self, x, y, fn):
        # x: (w, h, d, 1)
        # y: (w, h, d, c)
        y = aug_utils.combine_channels(y)
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
