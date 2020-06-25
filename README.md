# Python 3D Image Augmentation for 3D Image Segmentation
This is a libaray/framework containing a collection of mothods for 3D image data augmentation. Its intended usage is pair-wise (simultanous) augmentation of medical image data and their corresponding manual segmentation masks. It also supports non-pair-wise augmentation.

# Requriments
Install required packages: `pip install -r requirements.txt`

# Usage
- Open your segmentation masks and one-hot encode them. If you labels are not one-hot encoded, you may use the util function in `utils.to_channels` to convert a manual label of format (d1, d2, d3, 1) to (d1, d2, d3, n_classes).

- Reshape your images to the format (d1, d2, d3, 1).

- Apply augmentation as the following example:
```
from augmentor import Augmentor

# load data and mask
from augment3D import Augment3D

img1 = nib.load('test_img.nii.gz').get_fdata()            # img1.shape=[d1, d2, d3]
seg1 = nib.load('test_img_seg.nii.gz').get_fdata()        # seg1.shape=[d1, d2, d3]

# use the same image multiple times to simulate a batch
img1 = img1.reshape(img1.shape + (1,))                    # img1.shape=[d1, d2, d3, 1]
imgs = np.stack([img1, img1, img1, img1, img1], axis=0)   # imgs.shape=[n, d1, d2, d3, 1]

seg1 = seg1.reshape(seg1.shape + (1,))                    # seg1.shape=[d1, d2, d3, 1]
seg1 = to_channels(seg1)                                  # seg1.shape=[d1, d2, d3, n_classes]
seg = np.stack([seg1, seg1, seg1, seg1, seg1], axis=0)    # seg1.shape=[n, d1, d2, d3, n_classes]

# add singular operations to the augmentor
augmentor = Augment3D()
augmentor.add_elastic_deformation(8, [3, 4, 5, 6])
augmentor.add_affine_warp(30)
augmentor.add_uniaxial_swirl(3, 300)
augmentor.add_uniaxial_rotation(20)
augmentor.add_shifts([20, 20, 20])
augmentor.add_bezier_lut([0.3, 0.6], [0.3, 0.6], 0.45, degree=2)
augmentor.add_uniaxial_rotation(20)
augmentor.add_linear_gradient([3, 3, 3])

# add a complex augmentation sequence 
augmentor.add_sequence().add_uniaxial_rotation(20).add_shifts(
    [20, 20, 20]).add_elastic_deformation(3, [3, 4, 5]).end_sequence() # augmentation: rotation -> shifts -> deform

# list all the augmentations that have been added
augmentor.summary()

# use this in your training pipeline, alternatively, use it to generate training data before training
# every time this method is called, an added augmentation method is randomly chosen to apply augmentation
augmentor.apply_augmentation_to_batch(img, seg, debug=True)

```

Another usage example can be found in `test_augment.py`.
