import nibabel as nib
import numpy as np
from augmentor import Augmentor
from augmentation_utils import to_channels


if __name__ == "__main__":
    img1 = nib.load('test_img.nii.gz').get_fdata()
    seg1 = nib.load('test_img_seg.nii.gz').get_fdata()
    # use the same image multiple times to simulate a batch
    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1, img1, img1], axis=0)

    # seg1 = seg1.reshape(seg1.shape + (1,))
    seg1 = to_channels(seg1)
    seg = np.stack([seg1, seg1, seg1], axis=0)
    augmentor = Augmentor()
    augmentor.add_linear_gradient_fn([1, 1, 1])
    # augmentor.add_uniaxial_rotation_fn(22)
    # augmentor.add_shift_fn([30, 30, 30])
    # augmentor.add_elastic_deformation_fn(3, [3, 4, 5, 6])
    # augmentor.add_swirl_fn(1.2, 300)
    # augmentor.add_affine_warp_fn(20)
    # augmentor.add_sequence().add_rotation_fn(20).add_shift_fn(
    #     [20, 20, 20]).add_elastic_deformation_fn(3, [3, 4, 5]).end_sequence()

    # augmentor.add_sequence().add_rotation_fn(45).add_shift_fn(
    #     [50, 50, 50]).add_elastic_deformation_fn(8, [12, 16, 20]).end_sequence()

    augmentor.summary()
    # while True:
    augmentor.apply_augmentation_to_batch(img, seg, debug=True)
