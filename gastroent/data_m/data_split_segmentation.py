from pathlib import Path

from data_m.data_split_base import DataSplitBase

class DataSplitSegmentation(DataSplitBase):
    """
    Creates data splits specifically for image segmentation tasks.
    This subclass explicitly handles the pairing of 
    raw images with their corresponding masks. It groups them to prevent data 
    leakage, splits them into K-folds, and then flattens the folds back into 
    individual image-mask pairs ready for a PyTorch DataLoader.

    Supported datasets: 'Kvasir-SEG/kvasir-seg', 'Kvasir-SEG/kvasir-sessile', 
    and 'CVC-ClinicDB'.

    Args:
        data_dir (str): The name of the dataset folder (e.g., 'CVC-ClinicDB'). 
            Passed to base class.
        n_folds (int, optional): The number of cross-validation folds to create. 
            Defaults to 3. Passed to base class.
        seed (int, optional): Random seed for reproducible splitting. 
            Defaults to 42. Passed to base class.

    Raises:
        ValueError: If the provided `data_dir` is not in the list of supported datasets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.dataset_name not in {'Kvasir-SEG/kvasir-seg', 'Kvasir-SEG/kvasir-sessile', 'CVC-ClinicDB'}:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}. Supported datasets are: Kvasir-SEG/kvasir-seg, Kvasir-SEG/kvasir-sessile, and CVC-ClinicDB.')

        self.folds = self._get_folds()

    def _get_folds(self) -> list[dict[str, list[dict[str, Path]]]]:
        """
        Pairs images with masks, applies group-aware splitting, and flattens results.

        Locates all images and expects a corresponding mask with the exact same 
        filename in the 'masks' directory. Groups images based on 'split_rules.json', 
        generates the K-fold splits, and unwraps the groups into a flat list of 
        dictionaries for easy iteration during training.

        Returns:
            list[dict[str, list[dict[str, Path]]]]: A list of folds. Each fold is a 
                dictionary with 'train' and 'val' keys. The values are flat lists 
                of dictionaries containing 'image' and 'mask' Path objects.

        Raises:
            FileNotFoundError: If the 'images' or 'masks' directories are missing, 
                or if an image does not have a corresponding mask.
        """

        image_dir = self.source_dir / 'images'
        mask_dir = self.source_dir / 'masks'

        if not image_dir.exists():
            raise FileNotFoundError(f'Image directory not found: {image_dir}.')

        if not mask_dir.exists():
            raise FileNotFoundError(f'Mask directory not found: {mask_dir}.')

        images = self._get_image_paths(image_dir)
        image_mask_pairs = []

        for image in images:
            mask = mask_dir / image.name

            if not mask.exists():
                raise FileNotFoundError(f'Mask not found for image: {image}')

            image_mask_pairs.append({'image': image,
                                     'mask': mask})

        rules_path = self.source_dir / 'split_rules.json'
        image_groups = self._get_groups([pair['image'] for pair in image_mask_pairs], rules_path)
        image_to_pair = {pair['image']: pair for pair in image_mask_pairs}
        self.samples = [{'files': [image_to_pair[image] for image in group]} for group in image_groups]
        self.groups = image_groups
        folds = self._split(self.samples)

        # Flatten groups into individual images
        for fold in folds:
            fold['train'] = [item for group in fold['train'] for item in group['files']]
            fold['val'] = [item for group in fold['val'] for item in group['files']]

        return folds

# db = DataSplitSegmentation('CVC-ClinicDB')
# db.validate_folds()

# db = DataSplitSegmentation('Kvasir-SEG/kvasir-seg')
# db.validate_folds()

# db = DataSplitSegmentation('Kvasir-SEG/kvasir-sessile')
# db.validate_folds()