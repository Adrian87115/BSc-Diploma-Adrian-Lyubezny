from torchvision.transforms import v2
from torch.utils.data import Dataset
from typing import Any

class DatasetBase(Dataset):
    """
    Base class for image datasets.
    
    Args:
    dataset_name (str): Name of the used dataset.
    data (list): Data containing images and labels/masks.
    prep (dict[str, Any]): Preprocessing of image.
    rgb (bool, optional): Selection of RGB or Grayscale. Defaults to True.
    """

    def __init__(self, dataset_name: str, data: list, prep: dict[str, Any], rgb: bool = True):
        interpolation_map = {'bilinear': v2.InterpolationMode.BILINEAR,
                             'bicubic': v2.InterpolationMode.BICUBIC,
                             'nearest': v2.InterpolationMode.NEAREST}
        self.dataset_name = dataset_name
        self.data = data
        self.rgb = rgb
        self.to_image = v2.ToImage()
        self.resize = v2.Resize(size = prep['resize_size'], interpolation = interpolation_map[prep['interpolation_type']])
        self.mask_resize = v2.Resize(size = prep['resize_size'], interpolation = v2.InterpolationMode.NEAREST)
        self.crop = v2.CenterCrop(size = prep['center_crop'])

    def __len__(self) -> int:
        """
        Length of the data.

        Returns:
            Number of present samples in the dataset.
        """

        return len(self.data)