from pathlib import Path
from typing import Any

from data_m.data_split_base import DataSplitBase

class DataSplitClassification(DataSplitBase):
    """
    Creates data splits specifically for image classification tasks.
    This subclass traverses a dataset directory  to identify classes 
    based on leaf-node subdirectories. It extracts images, 
    applies group-aware splitting rules (per class), generates stratified K-folds, 
    and flattens the results for standard PyTorch classification DataLoaders.

    Supported datasets: 'Kvasir', 'HyperKvasir'.

    Args:
        data_dir (str): The name of the dataset folder (e.g., 'HyperKvasir'). 
            Passed to base class.
        n_folds (int, optional): The number of cross-validation folds to create. 
            Defaults to 3. Passed to base class.
        seed (int, optional): Random seed for reproducible splitting. 
            Defaults to 42. Passed to base class.

    Attributes:
        folds (list[dict[str, list[dict[str, Any]]]]): The generated cross-validation 
            folds containing 'train' and 'val' splits of image-label pairs.
        classes (list[str]): A sorted list of the extracted class names.
        samples (list[dict[str, Any]]): A list of grouped image-label pairs 
            before flattening.
        groups (list[list[Path]]): A list of the grouped image paths used to 
            prevent data leakage.

    Raises:
        ValueError: If the provided `data_dir` is not in the list of supported datasets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.data_dir not in {'Kvasir', 'HyperKvasir'}:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}. Supported datasets are: Kvasir, HyperKvasir.')

        self.folds = self._get_folds()

    def _get_all_classes(self) -> tuple[list[Path], dict[Path, list[Path]]]:
        """
        Identifies classification classes and retrieves their image paths.
        Traverses the dataset directory to find "leaf" directories (folders that 
        do not contain any other folders). Treats these directory names as class 
        labels and gathers all valid images within them.

        Returns:
            tuple[list[Path], dict[Path, list[Path]]]: A tuple containing:
                - A sorted list of Path objects representing the class directories.
                - A dictionary mapping each class directory Path to a list of its 
                  contained image Paths.
        """

        data = {}

        for class_dir in self.source_dir.rglob('*'):
            if not class_dir.is_dir():
                continue

            if any(child.is_dir() for child in class_dir.iterdir()):
                continue

            images = self._get_image_paths(class_dir)

            if not images:
                continue

            data[class_dir] = images

        class_dirs = sorted(data)
        self.classes = [path.name for path in class_dirs]
        return class_dirs, data

    def _get_folds(self) -> list[dict[str, list[dict[str, Any]]]]:
        """
        Applies grouping, stratifies by class label, and splits the data.

        Reads class-specific 'split_rules.json' files to group indivisible 
        samples. Passes the grouped data and class labels to the base class 
        splitter for Stratified Group K-Fold generation, ensuring balanced 
        classes across folds. Finally, flattens the groups into single 
        image-label dictionaries.

        Returns:
            list[dict[str, list[dict[str, Any]]]]: A list of folds. Each fold is a 
                dictionary with 'train' and 'val' keys. The values are flat lists 
                of dictionaries containing 'image' (Path) and 'label' (str).
        """

        class_dirs, data = self._get_all_classes()

        samples = []

        for class_dir in class_dirs:
            cls = class_dir.name
            rules_path = class_dir / 'split_rules.json'
            groups = self._get_groups(data[class_dir], rules_path)

            for group in groups:
                samples.append({'label': cls,
                                'image': group})

        self.samples = samples
        self.groups = [sample['image'] for sample in samples]
        labels = [sample['label'] for sample in samples]
        folds = self._split(samples, labels)

        # Flatten groups after splitting.
        for fold in folds:
            fold['train'] = [{'image': image,
                              'label': group['label']}
                             for group in fold['train']
                             for image in group['image']]

            fold['val'] = [{'image': image,
                            'label': group['label']}
                           for group in fold['val']
                           for image in group['image']]

        return folds

# db = DataSplitClassification('Kvasir')
# db.validate_folds()

# db = DataSplitClassification('HyperKvasir')
# db.validate_folds()