import nibabel as nib
import numpy as np
from augmentor import Augment3D
from utils import to_channels

path = "data/"
categorical = False

if __name__ == "__main__":
    img1 = nib.load(path+'case_4.nii.gz').get_fdata()
    seg1 = nib.load(path+'seg_4.nii.gz').get_fdata()
    # use the same image multiple times to simulate a batch
#    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1, img1, img1], axis=0)
    print("Batch Shape:", img.shape)

    # seg1 = seg1.reshape(seg1.shape + (1,))
    if categorical:
        seg1 = to_channels(seg1)
    seg = np.stack([seg1, seg1, seg1], axis=0)
    augmentor = Augment3D(categorical=categorical)
#    augmentor.add_elastic_deformation(8, [3, 4, 5, 6])
    # augmentor.add_elastic_deformation(3, [3, 4, 5, 6])
    augmentor.add_affine_warp(15)
#    augmentor.add_uniaxial_swirl(3, 300)
    augmentor.add_uniaxial_rotation(15)
    augmentor.add_shifts([20, 20, 20])
    # augmentor.add_bezier_lut([0.3, 0.6], [0.3, 0.6], 0.45, degree=2)
    # augmentor.add_bezier_lut([0.3, 0.6], [0.3, 0.6], 0.15, degree=2)
    # augmentor.add_uniaxial_rotation(20)
    # augmentor.add_sequence().add_uniaxial_rotation(20).add_shifts(
    #     [20, 20, 20]).add_elastic_deformation(3, [3, 4, 5]).end_sequence()

    # augmentor.add_linear_gradient([3, 3, 3])
    augmentor.summary()

    augImg, augSeg = augmentor.apply_augmentation_to_batch(img, seg, debug=True)
    
    for index, (image, segment) in enumerate( zip(augImg, augSeg) ):
        testImage = nib.Nifti1Image(image, np.eye(4))
        testImageName = path+"result_"+str(index)+".nii.gz"
        testImage.to_filename(testImageName)
        testSeg = nib.Nifti1Image(segment, np.eye(4))
        testSegName = path+"result_seg_"+str(index)+".nii.gz"
        testSeg.to_filename(testSegName)

print("Complete")