import random
import nibabel as nib
import os
import numpy as np
from utils import combine_channels, normalize
from augmentations.rotation import Rotation
from augmentations.shifts import Shifts
from augmentations.swirl import Swirl
from augmentations.elastic_deformation import ElasticDeformation
from augmentations.affline_warp import AffineWarp
from augmentations.gradient import LinearGradient
from augmentations.lut import BezierLUT
from augmentations.sequence_queue import SequenceQueue
from augmentations.resize import Rescale, RandomCrop


class Augmentor:
    """
    Apply random augmentations to images. Parameters are sampled at random from truncated normal distributions!
    NOTE: all normal distributions used in this work are truncted norms!
    """

    def __init__(self, normalise_x=True, normalise_y=False):
        self._augmentations = []
        self._sequence_queue = None
        self.normalise_x = normalise_x
        self.normalise_y = normalise_y

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

    def apply_augmentation_to_sample(self, x, y, copy=True, debug=False):
        if copy:
            x = x.copy()
            y = y.copy()
        if len(self._augmentations) == 0:
            raise ValueError(
                "The augmentor does not have any augmentation function - add augmentation functions before applying augmentation.")
        x, y = self._do_aug(x, y)
        if debug:
            self._save_debug_img(x, y, 0)
        return x, y

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
            x, y = self._do_aug(x, y)
            if debug:
                self._save_debug_img(x, y, i)
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
        """
        Perform random uniaxial rotation. rotation axes are choosing at random (1 of the 3 dimensions).

        Args:
            std (float): standard dividation used for sampling rotation angles (in degrees). 
                Rotation angles are sampled from a normal distribution N~(0, std^2)
        """
        return self._augmentation_handler(Rotation(std))

    def add_shifts(self, shift_stds):
        """
        Perform random shifts along on or more axes.

        Args:
            shift_stds ([float, float, float]): a list of 3 floats. Shift magnitudes (in pixels) are sampled from a normal distribution N~(0, std^2)
                - shift_stds[0]: standard dividation used for sampling shifts along the first dimension of the image,
                - shift_stds[1]: standard dividation used for sampling shifts along the second dimension of the image,
                - shift_stds[2]: standard dividation used for sampling shifts along the third dimension of the image.
        """
        return self._augmentation_handler(Shifts(shift_stds))

    def add_rescale(self, std):
        return self._augmentation_handler(Rescale(std))

    def add_patch_crop(self, std):
        return self._augmentation_handler(RandomCrop(std))

    def add_uniaxial_swirl(self, strength_std, radius):
        """
        Perform uniaxial swirl. Swirls are symmetrical and centred.

        Args:
            strength_std (float): standard dividation used for sampling swirl strengths. Swirl strengths are sampled from a normal 
                distribution N~(0, std^2) (recommend using debug mode to inspect the effect should usually be a small number less than 10). 
            radius (int): swirl radius -  this is a fixed value and there is no randomness involved.
        """
        return self._augmentation_handler(Swirl(strength_std, radius))

    def add_elastic_deformation(self, sigma_std, possible_points):
        """
        Perform random elastic deform.
        Args:
            sigma_std (float): standard deviation of the normal distribution (affects deformation strengths)
            possible_points ([int, int, int...]): list of integers > 0. The Augmentor will randomly select an integer N from the list
                and apply deformations at N random points in the image.

        """
        return self._augmentation_handler(ElasticDeformation(sigma_std, possible_points))

    def add_affine_warp(self, vertex_percentage_std):
        """
        Perform random affine warping. This operation uses the original image's 4 corners (v1, v2, v3, v4) as the source vertices, shifts are applied to these vertices to acquire the destination vertices (v1', v2', v3', v4'). We then apply affline transform to map (v1, v2, v3, v4) to (v1', v2', v3', v4')

        Args:
            vertex_percentage_std (float): standard dividation used for sampling shifts for each vertex.
                for each dimension of v:
                    percentage ~ N(0, vertex_percentage_std^2)
                    v' = v + (1 + percentage)
        """
        return self._augmentation_handler(AffineWarp(vertex_percentage_std))

    def add_linear_gradient(self, gradient_stds):
        # Work in progress -> need to validate effectiveness
        assert len(gradient_stds) == 3
        return self._augmentation_handler(LinearGradient(gradient_stds))

    def add_bezier_lut(self, xs, ys, y_std, degree=2):
        # Work in progress -> need to validate effectiveness
        assert len(xs) == len(ys)
        return self._augmentation_handler(BezierLUT(xs, ys, y_std, degree))

    def _do_aug(self, x, y):
        aug = random.choice(self._augmentations)
        x, y = aug.execute(x, y)
        if self.normalise_x:
            x = normalize(x)
        if self.normalise_y:
            y = normalize(y)

        return x, y

    def summary(self):
        print("=" * 15 + " Augmentation Functions " + "=" * 15)
        [print(aug) for aug in self._augmentations]

    def _save_debug_img(self, x, y, idx):
        # x: (w, h, d, 1)
        # y: (w, h, d, c)
        y = combine_channels(y)
        x = nib.Nifti1Image(x, np.eye(4))
        y = nib.Nifti1Image(y, np.eye(4))
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        nib.save(x, os.path.join(debug_dir, "%d.nii" % idx))
        nib.save(y, os.path.join(debug_dir, "%d_seg.nii" % idx))
