class BaseAugmentation:
    def execute(self, image, mask):
        raise NotImplementedError
