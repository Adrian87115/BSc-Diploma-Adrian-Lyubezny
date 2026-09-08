from torchvision.transforms import v2
from torch.utils.data import Dataset

class DatasetBase(Dataset):
    """
    Base class for image datasets.
    
    Args:
    dataset_name (str): Name of the used dataset.
    data (list): Data containing images and labels/masks.
    rgb (bool, optional): Selection of RGB or Grayscale. Defaults to True.
    """

    def __init__(self, dataset_name: str, data: list, rgb: bool = True):
        self.dataset_name = dataset_name
        self.data = data
        self.rgb = rgb
        self.to_image = v2.ToImage()

    def __len__(self) -> int:
        """
        Length of the data.

        Returns:
            Number of present samples in the dataset.
        """

        return len(self.data)