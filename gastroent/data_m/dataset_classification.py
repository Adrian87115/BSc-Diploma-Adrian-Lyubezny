from PIL import Image
import torch

from data_m.dataset_base import DatasetBase

class DatasetClassification(DatasetBase):
    """
    PyTorch Dataset for image classification.

    Inherits from DatasetBase. Responsible for loading images from disk, 
    applying the required transforms, and mapping string class names to 
    integer tensor labels for training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_unique_labels(self) -> dict[str, int]:
        """
        Identifies all unique classes present in the current data split.

        Returns:
            dict[str, int]: A dictionary mapping the class names found in 
                `self.data` to their corresponding integer indices based on 
                the global CLASS_TO_IDX mapping.
        """

        unique_labels = set(sample['label'] for sample in self.data)
        return {label: CLASS_TO_IDX[label] for label in unique_labels}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches and preprocesses the image and label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The transformed image tensor (C, H, W).
                - The class label as a 0-dimensional torch.long tensor.
        """

        sample = self.data[index]
        image = Image.open(sample['image']).convert('RGB' if self.rgb else 'L')
        
        if self.transforms is not None:
            image = self.transforms(image)

        label = torch.tensor(CLASS_TO_IDX[sample['label']], dtype = torch.long)
        return image, label

CLASS_TO_IDX = {# Kvasir #
                'dyed-lifted-polyps': 0,
                'dyed-resection-margins': 1,
                'esophagitis': 2,   # Is conflicting with more detailed 'esophagitis' classes present in HyperKvasir
                'normal-cecum': 3,
                'normal-pylorus': 4,
                'normal-z-line': 5,
                'polyps': 6,
                'ulcerative-colitis': 7,    # Is conflicting with more detailed 'ulcerative-colitis' classes present in HyperKvasir
                # HyperKvasir #
                'cecum': 3,
                'ileum' : 8,
                'retroflex-rectum': 9,
                'hemorrhoids': 10,
                # polyps
                'ulcerative-colitis-grade-0-1': 11,
                'ulcerative-colitis-grade-1': 12,
                'ulcerative-colitis-grade-1-2': 13,
                'ulcerative-colitis-grade-2': 14,
                'ulcerative-colitis-grade-2-3': 15,
                'ulcerative-colitis-grade-3': 16,
                'bbps-0-1': 17,
                'bbps-2-3': 18,
                'impacted-stool': 19,
                # dyed-lifted-polyps
                # dyed-resection-margins
                'pylorus': 4,
                'retroflex-stomach': 20,
                'z-line': 5,
                'barretts': 21,
                'barretts-short-segment': 22,
                'esophagitis-a': 23,
                'esophagitis-b-d': 24}