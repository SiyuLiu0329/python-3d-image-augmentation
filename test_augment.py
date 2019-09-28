import nibabel as nib
import numpy as np
from augment3D import Augment3D
from utils import to_channels

path = "MPRAGE_x2/"
path_seg = "seg_x2/"
output_path = "output/"
categorical = False

if __name__ == "__main__":
    img1 = nib.load(path+'case_4.nii.gz').get_fdata()
    seg1 = nib.load(path_seg+'seg_4.nii.gz').get_fdata()
    # use the same image multiple times to simulate a batch
    img1 = img1.reshape(img1.shape + (1,))
    img = np.stack([img1, img1, img1], axis=0)
    print("Batch Shape:", img.shape)

    seg1 = seg1.reshape(seg1.shape + (1,))
    if categorical:
        seg1 = to_channels(seg1)
    seg = np.stack([seg1, seg1, seg1], axis=0)
    augmentor = Augment3D(categorical=categorical)
#    augmentor.add_elastic_deformation(8, [3, 4, 5, 6])
    augmentor.add_elastic_deformation(3, [3, 4, 5, 6])
    augmentor.add_affine_warp(10)
#    augmentor.add_uniaxial_swirl(3, 300)
    augmentor.add_uniaxial_rotation(15)
    augmentor.add_shifts([10, 10, 10])
    # augmentor.add_uniaxial_rotation(20)
    augmentor.summary()

    augImg, augSeg = augmentor.apply_augmentation_to_batch(img, seg, debug=True)
    
    for index, (image, segment) in enumerate( zip(augImg, augSeg) ):
        testImage = nib.Nifti1Image(image, np.eye(4))
        testImageName = output_path+"result_"+str(index)+".nii.gz"
        testImage.to_filename(testImageName)
        testSeg = nib.Nifti1Image(segment, np.eye(4))
        testSegName = output_path+"result_seg_"+str(index)+".nii.gz"
        testSeg.to_filename(testSegName)

print("Complete")