import nibabel as nib
import numpy as np
from augmentation import apply_augmentation, to_channels


if __name__ == "__main__":
    img1 = nib.load('3d-MRI-scan.nii.gz').get_fdata()
    seg1 = nib.load(
        '3d-MRI-segmentation.nii.gz').get_fdata()

    # use the same image multiple times to simulate a batch
    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1, img1, img1], axis=0)

    # seg1 = seg1.reshape(seg1.shape + (1,))
    seg1 = to_channels(seg1)
    seg = np.stack([seg1, seg1, seg1], axis=0)
    while True:
        apply_augmentation(img, seg, debug=True)
