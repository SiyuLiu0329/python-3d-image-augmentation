import random
import matplotlib.pyplot as plt
from utils import combine_channels, normalize
from augmentations.rotation import Rotation
from augmentations.shifts import Shifts
from augmentations.swirl import Swirl
from augmentations.elastic_deformation import ElasticDeformation
from augmentations.affline_warp import AffineWarp
from augmentations.gradient import LinearGradient
from augmentations.lut import BezierLUT
from augmentations.sequence_queue import SequenceQueue


class Augmentor:
    def __init__(self):
        self._augmentations = []
        self._sequence_queue = None
        self.normalise_x = True
        self.normalise_y = False

    def add_sequence(self):
        assert self._sequence_queue is None
        self._sequence_queue = SequenceQueue()
        return self

    def end_sequence(self):
        sequence = self._sequence_queue
        assert sequence is not None
        assert len(sequence) != 0
        self._augmentations.append(self._sequence_queue)
        self._sequence_queue = None
        return self

    def apply_augmentation_to_batch(self, x_batch, y_batch, copy=True, debug=False):
        # x: (w, h, d, 1)
        # y: (w, h, d, c)
        if copy:
            x_batch = x_batch.copy()
            y_batch = y_batch.copy()

        if len(self._augmentations) == 0:
            raise ValueError(
                "The augmentor does not have any augmentation function - add augmentation functions before applying augmentation.")
        assert len(x_batch) == len(y_batch)
        for i in range(len(x_batch)):
            x, y = x_batch[i], y_batch[i]
            x, y = self._do_aug(x, y, debug=debug)
            x_batch[i] = x
            y_batch[i] = y

        return x_batch, y_batch,

    def _augmentation_handler(self, operation):
        if self._sequence_queue is not None:
            self._sequence_queue.enque(operation)
        else:
            self._augmentations.append(operation)
        return self

    def add_uniaxial_rotation(self, std):
        return self._augmentation_handler(Rotation(std))

    def add_shifts(self, shift_stds):
        return self._augmentation_handler(Shifts(shift_stds))

    def add_uniaxial_swirl(self, strength_std, radius):
        return self._augmentation_handler(Swirl(strength_std, radius))

    def add_elastic_deformation(self, sigma_std, possible_points):
        return self._augmentation_handler(ElasticDeformation(sigma_std, possible_points))

    def add_affine_warp(self, vertex_percentage_std):
        return self._augmentation_handler(AffineWarp(vertex_percentage_std))

    def add_linear_gradient(self, gradient_stds):
        assert len(gradient_stds) == 3
        return self._augmentation_handler(LinearGradient(gradient_stds))

    def add_bezier_lut(self, xs, ys, degree=2):
        return self._augmentation_handler(BezierLUT(xs, ys, degree))

    def _do_aug(self, x, y, debug=False):
        aug = random.choice(self._augmentations)
        x, y = aug.execute(x, y)
        if debug:
            self._debug_show(x, y, aug)

        if self.normalise_x:
            x = normalize(x)
        if self.normalise_y:
            y = normalize(y)

        return x, y

    def summary(self):
        print("=" * 15 + " Augmentation Functions " + "=" * 15)
        [print(aug) for aug in self._augmentations]

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
