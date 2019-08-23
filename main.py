import nibabel as nib
import numpy as np
from augmentor import Augmentor
from utils import to_channels


if __name__ == "__main__":
    img1 = nib.load('test_img.nii.gz').get_fdata()
    seg1 = nib.load('test_img_seg.nii.gz').get_fdata()
    # use the same image multiple times to simulate a batch
    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1, img1, img1, img1, img1], axis=0)

    # seg1 = seg1.reshape(seg1.shape + (1,))
    seg1 = to_channels(seg1)
    seg = np.stack([seg1, seg1, seg1, seg1, seg1], axis=0)
    augmentor = Augmentor()
    augmentor.add_elastic_deformation(8, [3, 4, 5, 6])
    # augmentor.add_elastic_deformation(3, [3, 4, 5, 6])
    # augmentor.add_affine_warp(20)
    # augmentor.add_uniaxial_swirl(2, 300)
    # augmentor.add_uniaxial_rotation(20)
    # augmentor.add_shifts([20, 20, 20])
    # augmentor.add_bezier_lut([0.3, 0.6], [0.3, 0.6], 0.45, degree=2)
    # augmentor.add_bezier_lut([0.3, 0.6], [0.3, 0.6], 0.15, degree=2)
    # augmentor.add_uniaxial_rotation(20)
    # augmentor.add_sequence().add_uniaxial_rotation(20).add_shifts(
    #     [20, 20, 20]).add_elastic_deformation(3, [3, 4, 5]).end_sequence()

    # augmentor.add_linear_gradient([3, 3, 3])
    augmentor.summary()

    augmentor.apply_augmentation_to_batch(img, seg, debug=True)
