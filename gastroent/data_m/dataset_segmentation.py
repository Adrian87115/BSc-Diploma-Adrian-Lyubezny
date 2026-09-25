from PIL import Image
import numpy as np
import torch

from data_m.dataset_base import DatasetBase

class DatasetSegmentation(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches and preprocesses the image and mask at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The transformed image tensor (C, H, W).
                - The binary mask tensor (1, H, W) where background is 0 and target is 1.
        """

        sample = self.data[index]
        image = Image.open(sample['image']).convert('RGB' if self.rgb else 'L')
        mask = Image.open(sample['mask']).convert('L')
        
        image = self.to_image(image)
        mask = self.to_image(mask)

        image = self.resize(image)
        mask = self.mask_resize(mask)

        image = image.float() / 255.0
        mask = (mask > 0).float().long()    # black -> 0, white -> 1
        return image, mask