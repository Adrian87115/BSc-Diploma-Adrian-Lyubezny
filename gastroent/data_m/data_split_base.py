import json
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from typing import Any

from utils import ALLOWED_EXTENSIONS

class DataSplitBase:
    """
    A base class for obtaining and splitting image file paths.

    Obtains all paths and groups images according to a set of rules (if present). 
    Splits the data into K-folds using stratification to maintain class balance.

    Args:
        data_dir (str): The name of the dataset folder located inside the 
            project's 'data' directory (e.g., 'Kvasir').
        n_folds (int, optional): The number of cross-validation folds to create. 
            Defaults to 3.
        seed (int, optional): Random seed for reproducible splitting. Defaults to 42.

    Raises:
        FileNotFoundError: If the dataset directory is not found in the 'data' folder.
        NotADirectoryError: If the resolved dataset path points to a file instead 
            of a directory.
    """

    def __init__(self, data_dir: str, n_folds: int = 3, seed: int = 42):
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_dir = self.project_root / 'data'

        self.data_dir = data_dir
        self.n_folds = n_folds
        self.seed = seed

        self.dataset_name = data_dir
        self.source_dir = self.raw_dir / data_dir

        if not self.source_dir.exists():
            raise FileNotFoundError(f'Dataset not found: {self.source_dir}.')

        if not self.source_dir.is_dir():
            raise NotADirectoryError(f'Expected a directory, but found a file: {self.source_dir}.')

    def _get_image_paths(self, search_dir: Path | str) -> list[Path]:
        """
        Obtains image files from the given directory recursively.

        Args:
            search_dir (Path | str): The directory to search for images.

        Returns:
            list[Path]: A list of paths to all valid image files.
        """
        search_path = Path(search_dir)
        return [path for path in search_path.rglob('*') if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS]

    def _get_groups(self, files: list[Path], rules_path: Path | str | None = None) -> list[list[Path]]:
        """
        Converts files into indivisible groups to avoid data leakage.

        Files listed together in the JSON rules file are grouped into the same list. 
        Files not listed in the JSON (or if no rules are provided) are treated as 
        independent, single-item groups.

        Args:
            files (list[Path]): A list of image file paths to be grouped.
            rules_path (Path | str | None, optional): Path to a JSON file containing 
                split rules. The JSON should be a dictionary where values are lists 
                of filenames belonging to the same group. Defaults to None.
                (e.g., {"group1": ["1.png", ...], ...})

        Returns:
            list[list[Path]]: A list of file groups, where each group is a list of Paths.
        """

        self.groups = []
        grouped_files = set()
        file_map = {path.name: path for path in files}

        if rules_path and Path(rules_path).exists():
            with open(rules_path, 'r', encoding = 'utf-8') as f:
                split_rules = json.load(f)

            for group in split_rules.values():
                group_files = [file_map[name] for name in group if name in file_map]

                if group_files:
                    self.groups.append(group_files)
                    grouped_files.update(group_files)

        # Single and not mentioned in rules files.
        for path in files:
            if path not in grouped_files:
                self.groups.append([path])

        return self.groups

    def _split(self, samples: list[dict[str, Any]], labels: list[int] | None = None) -> list[dict[str, list[dict[str, Any]]]]:
        """
        Creates group-aware K-fold splits to prevent data leakage.

        If labels are provided (Classification), uses StratifiedGroupKFold to 
        maintain class balance. If not (Segmentation), uses standard GroupKFold.

        Args:
            samples (list[dict[str, Any]]): A list of dictionaries, where each dict 
                represents an indivisible group of images (e.g., sequence of images).
            labels (list[int] | None, optional): A list of class labels corresponding 
                to each group. Required for stratification. Defaults to None.

        Returns:
            list[dict[str, list[dict[str, Any]]]]: A list of folds. Each fold is a 
                dictionary with 'train' and 'val' keys containing the respective samples.

        Raises:
            ValueError: If `n_folds` < 2, or if there are fewer samples than folds.
        """

        if self.n_folds < 2:
            raise ValueError('n_folds must be >= 2.')

        if len(samples) < self.n_folds:
            raise ValueError(f'Cannot create {self.n_folds} folds from {len(samples)} samples.')

        # Samples are already in the form of groups, they are treated as inseparable entities.
        groups = list(range(len(samples)))

        if labels is not None:
            splitter = StratifiedGroupKFold(n_splits = self.n_folds, shuffle = True, random_state = self.seed)
            splits = splitter.split(X = samples, y = labels, groups = groups)

        else:
            splitter = GroupKFold(n_splits = self.n_folds)
            splits = splitter.split(X = samples, groups = groups)

        folds = []

        for train_idx, val_idx in splits:
            folds.append({'train': [samples[i] for i in train_idx],
                          'val': [samples[i] for i in val_idx]})

        return folds

    def validate_folds(self) -> None:
        """
        Validates the integrity of the generated data folds.

        Checks:
        1. No overlap between training and validation sets.
        2. No files are lost.
        3. Pre-defined groups remain intact.
        4. Every file appears in validation exactly once.
        """

        print(f'Validation {self.dataset_name}')

        all_files = {_get_id(image) for group in self.groups for image in group}
        validation_seen = set()

        for i, fold in enumerate(self.folds):
            train_files = {_get_id(sample['image']) for sample in fold['train']}
            val_files = {_get_id(sample['image']) for sample in fold['val']}

            train_groups = sum(any(_get_id(image) in train_files for image in group) for group in self.groups)
            val_groups = sum(any(_get_id(image) in val_files for image in group) for group in self.groups)

            # No overlap between train and validation.
            assert not train_files & val_files, f'Fold {i}: train/val overlap!'

            # No files missing.
            assert train_files | val_files == all_files, f'Fold {i}: files missing!'

            # Groups must remain intact.
            for group in self.groups:
                group_files = {_get_id(path) for path in group}

                in_train = group_files & train_files
                in_val = group_files & val_files

                assert not (in_train and in_val), f'Fold {i}: group was split! {group_files}'

            # Every file must appear in validation exactly once.
            assert not validation_seen & val_files, f'Fold {i}: files already appeared in validation!'

            validation_seen.update(val_files)
            print(f'Fold {i}:\n'
                  f'  Train: {len(train_files)} images | {train_groups} groups\n'
                  f'  Val:   {len(val_files)} images | {val_groups} groups')

        assert validation_seen == all_files, 'Not every file appeared in validation exactly once.'

        print('All fold checks passed.\n')

def _get_id(item: dict[str, Any] | str | Path) -> str:
    """
    Helper function to extract the image ID as a string.

    Args:
        item (dict[str, Any] | str | Path): The data item, which can be a 
            dictionary containing an 'image' key, or a direct file path/string.

    Returns:
        str: The string representation of the image path or ID.
    """

    return str(item['image']) if isinstance(item, dict) else str(item)