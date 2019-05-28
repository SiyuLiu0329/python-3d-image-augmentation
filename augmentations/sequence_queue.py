from augmentations.base_augmentation import BaseAugmentation


class SequenceQueue(BaseAugmentation):
    def __init__(self):
        self.operations = []

    def enque(self, operation):
        self.operations.append(operation)

    def __len__(self):
        return len(self.operations)

    def execute(self, image, mask):
        for operation in self.operations:
            image, mask = operation.execute(image, mask)
        return image, mask
