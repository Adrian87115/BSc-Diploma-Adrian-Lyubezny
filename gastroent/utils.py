import random
import numpy as np
import torch
import torch.distributed as dist
import os

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

def set_seeds(seed: int) -> None:
    """
    Sets seed for all used libraries to ensure reproducibility.

    Args:
        seed (int): Seed to reproduce used conditions.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup(world_size: int, rank: int, backend: str = 'gloo') -> None:
    """
    Sets up the PyTorch distributed training environment.

    Args:
        world_size (int): The total number of available GPUs.
        rank (int): ID of the current GPU.
        backend (str): Mode of the environment. The distributed backend. 'gloo' is standard for 
            Windows or CPU training, 'nccl' is standard for Linux multi-GPU. Defaults to 'gloo'.
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend, rank = rank, world_size = world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup() -> None:
     """
     Cleans up and destroys the distributed training environment.
     """
     
     dist.destroy_process_group()