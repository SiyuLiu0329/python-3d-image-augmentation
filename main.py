import nibabel as nib
import numpy as np
from augment3D import Augment3D
from utils import to_channels


if __name__ == "__main__":
    img1 = nib.load('testimg.nii.gz').get_fdata()
    seg1 = nib.load('testseg.nii.gz').get_fdata()
    # use the same image multiple times to simulate a batch
    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1], axis=0)

    # seg1 = seg1.reshape(seg1.shape + (1,))
    seg1 = to_channels(seg1)
    seg = np.stack([seg1], axis=0)
    augmentor = Augment3D()
    # augmentor.add_rescale(30)
    # augmentor.add_patch_crop(30)
    # augmentor.add_elastic_deformation(10, [3, 4, 5, 6])
    # augmentor.add_affine_warp(20)
    # augmentor.add_identity()
    # augmentor.add_uniaxial_swirl(3, 300)
    # augmentor.add_uniaxial_rotation(20)
    # augmentor.add_shifts([20, 20, 20])
    # augmentor.add_bezier_lut([0.3, 0.6], [0.3, 0.6], 0.45, degree=2)
    # augmentor.add_linear_gradient([3, 3, 3])

    augmentor.add_sequence().add_uniaxial_rotation(20).add_rescale(30).add_shifts(
        [20, 20, 20]).add_elastic_deformation(3, [3, 4, 5]).end_sequence()

    augmentor.summary()

    augmentor.apply_augmentation_to_sample(img1, seg1, debug=True)
