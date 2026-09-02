from PIL import Image
from torchvision import tv_tensors
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
                - The binary mask tensor (H, W) where background is 0 and 
                  target is 1.
        """

        sample = self.data[index]
        image = Image.open(sample['image']).convert('RGB' if self.rgb else 'L')
        mask = Image.open(sample['mask']).convert('L') 
        mask = tv_tensors.Mask(mask)    # Wraps the mask so torchvision.transforms.v2 knows how to handle it

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        mask = (mask.squeeze(0) > 0).long() # (1, H, W) -> (H, W) and black -> 0, white -> 1
        return image, mask