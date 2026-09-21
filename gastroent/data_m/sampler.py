from torch.utils.data import Sampler
import math
import torch
from collections import Counter, defaultdict

from data_m.dataset_base import DatasetBase

class DistributedRepeatedSampler(Sampler):
    """
    Custom implementation of sampler, merging functionality of distributed sampler with oversampling.
    Performs global oversampling, and then distributed partitioning.

    Args:
        dataset (DatasetBase): Data.
        max_repeats (int, optional): Maximum number of allowed repetitions for an image. Defaults to 10.
        num_replicas (int, optional): Number of processes participating in distributed training.
            Defaults to 1.
        rank (int, optional): Rank of the current process. Defaults to 0.
        shuffle (bool, optional): If True, sampler will shuffle indices. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler. Defaults to 42.
        drop_last (bool, optional): If True, then the sampler will drop the tail of the data 
            to make it evenly divisible across the number of replicas. 
            If False, the sampler will add extra indices to make the data evenly divisible across the replicas. 
            Defaults to False.
    """

    def __init__(self, dataset: DatasetBase, max_repeats: int = 10, num_replicas: int = 1, rank: int = 0, 
                 shuffle: bool = True, seed: int = 42, drop_last: bool = False):
        self.dataset = dataset
        self.max_repeats = max_repeats
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.epoch = 0
        self.num_samples = 0
        self.total_size = 0

    def _expand(self, indices: list[int], classes: list[int], generator: torch.Generator) -> list[int]:
        """
        Expands the dataset by performing oversampling.

        Args:
            indices (list[int]): List of all indices.
            classes (list[int]): List of classes, for the samples at the corresponding indices.
            generator (torch.Generator): Generator used in shuffling and sampling.
        
        Returns:
            list[int] -> Expanded list of indices.
        """

        counts_per_class = Counter(classes)
        upper_bound = max(counts_per_class.values())

        expansion_per_class = {cls: min(upper_bound, count * self.max_repeats) for cls, count in counts_per_class.items()}

        indices_by_class = defaultdict(list)

        for idx, cls in zip(indices, classes):
            indices_by_class[cls].append(idx)

        expanded_indices = []

        for cls, target_count in expansion_per_class.items():
            class_indices = indices_by_class[cls]
            class_count = len(class_indices)

            full_repeats = target_count // class_count
            remainder = target_count % class_count

            sampled = class_indices * full_repeats

            if remainder > 0:
                perm = torch.randperm(class_count, generator = generator)
                sampled.extend(class_indices[i] for i in perm[:remainder].tolist())

            expanded_indices.extend(sampled)

        if self.shuffle:
            perm = torch.randperm(len(expanded_indices), generator = generator)
            expanded_indices = [expanded_indices[i] for i in perm.tolist()]

        if self.drop_last:
            self.num_samples = len(expanded_indices) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(expanded_indices) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

        return expanded_indices

    def __iter__(self):
        """
        Generates an iterator over sample indices for the current process rank.

        Yields:
            Iterator[int]: An iterator yielding dataset indices assigned to the current worker rank.

        Raises:
            AssertionError: If the adjusted index pool length does not match ``self.total_size``.
        """

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator = generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        classes = [self.dataset.data[i]['label'] for i in indices]
        indices = self._expand(indices, classes, generator)

        if not self.drop_last:
            padding_size = self.total_size - len(indices)

            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]

        if len(indices) != self.total_size:
            raise AssertionError(f'Number of indices: {len(indices)}, does not match total_size: {self.total_size}.')

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        """
        Number of samples processed by this replica per epoch.
        """

        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Setting the epoch, for reshuffling between epochs.

        Args:
            epoch (int): Current epoch.
        """

        self.epoch = epoch