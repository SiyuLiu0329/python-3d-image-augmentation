from augmentations.base_augmentation import BaseAugmentation


class Identity(BaseAugmentation):
    def __init__(self):
        super().__init__()

    def execute(self, image, mask):
        return image, mask
