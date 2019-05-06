import nibabel as nib
import numpy as np
from augmentor import Augmentor
from augmentation_utils import to_channels


if __name__ == "__main__":
    img1 = nib.load('crop/PAT001_HIRF_Prisma_20171122_scan.nii.gz').get_fdata()
    seg1 = nib.load(
        'crop/PAT001_HIRF_Prisma_20171122_segment.nii.gz').get_fdata()

    # use the same image multiple times to simulate a batch
    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1, img1, img1], axis=0)

    # seg1 = seg1.reshape(seg1.shape + (1,))
    seg1 = to_channels(seg1)
    seg = np.stack([seg1, seg1, seg1], axis=0)
    augmentor = Augmentor()
    # augmentor.add_rotation_fn(20)
    # augmentor.add_shift_fn([20, 20, 20])
    # augmentor.add_elastic_deformation_fn(3, [3, 4, 5])
    # augmentor.add_swirl_fn(1, 300)

    while True:
        augmentor.apply_augmentation_to_batch(img, seg, debug=True)
