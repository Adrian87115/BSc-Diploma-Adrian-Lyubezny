import random
import numpy as np
import torch
import torch.distributed as dist
import os
import json
from pathlib import Path
from typing import Any
from torchvision.transforms import v2
from typing import Iterator

from models_m.classification_models import ClassificationModel, ResNetModel, ConvNeXtModel, SwinTransformerModel
from models_m.segmentation_models import SegmentationModel, UNetPlusPlusModel, DeepLabV3PlusModel, SegFormerModel
from data_m.data_split_classification import DataSplitClassification
from data_m.data_split_segmentation import DataSplitSegmentation
from data_m.dataset_classification import DatasetClassification, DatasetBase
from data_m.dataset_segmentation import DatasetSegmentation

# Allowed extensions of images and masks
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')  

# Required keys for building and running classification tasks
REQUIRED_SETUP_CLASSIFICATION = ['in_channels', 'num_classes', 'm_type',
                                 'pretrained', 'num_workers', 'persistent_workers',
                                 'prefetch_factor', 'pin_memory', 'model',
                                 'data_dir', 'n_folds', 'used_fold',
                                 'loss_function', 'optimizer', 'scheduler', 'batch_size',
                                 'num_epochs', 'save_freq', 'eval_freq']

# Required keys for building and running segmentation tasks
REQUIRED_SETUP_SEGMENTATION = ['in_channels', 'encoder', 'pretrained',
                               'weights_ver', 'num_workers', 'persistent_workers',
                               'prefetch_factor', 'pin_memory', 'model',
                               'data_dir', 'n_folds', 'used_fold',
                               'loss_function', 'optimizer', 'scheduler', 'batch_size',
                               'num_epochs', 'save_freq', 'eval_freq']

# Required keys for preprocessing
REQUIRED_PREP = ['resize_size', 'interpolation_type', 'center_crop', 'mean', 'std']

# Keys that can be used for augmentation
POSSIBLE_AUG = []

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

def get_config(config_path: str | Path, classification: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Obtains and validates the configuration of the experiment.

    Args:
        config_path (str | Path): Configuration json with configuration of the experiment.
        classificaion (bool): Information whether config belongs to classification,
            or to segmentaiton. Required to decide which keys are reuqired in the 
            configuration file.

    Returns:
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: setup for the training initialization,
            preprocessing configuration, and augmentaiton configuration.
    """

    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = json.load(f)
        setup = config['setup']
        prep = config['prep']
        aug = config['aug']

    required_keys = REQUIRED_SETUP_CLASSIFICATION if classification else REQUIRED_SETUP_SEGMENTATION

    setup = validate_setup(setup, required_keys)
    prep = validate_prep(prep, REQUIRED_PREP)
    aug = validate_aug(aug, POSSIBLE_AUG)

    return setup, prep, aug

def validate_setup(setup: dict[str, Any], required_keys: list[str]) -> dict[str, Any]:
    """
    Checks and validates the setup configuration.

    Args:
        setup (dict[str, Any]): Configuration dictionary.
        required_keys (list[str]): Required setup keys.

    Returns:
        dict[str, Any]: Validated setup dictionary.

    Raises:
        ValueError: If the configuration is invalid.
    """

    missing_keys = [key for key in required_keys if key not in setup]

    if missing_keys:
        raise ValueError(f'Missing obligatory setup variables: {missing_keys}')

    available_models = {'ResNetModel': ResNetModel,
                        'ConvNeXtModel': ConvNeXtModel,
                        'SwinTransformerModel': SwinTransformerModel,
                        'UNetPlusPlusModel': UNetPlusPlusModel,
                        'DeepLabV3PlusModel': DeepLabV3PlusModel,
                        'SegFormerModel': SegFormerModel}

    if isinstance(setup['model'], str):
        if setup['model'] not in available_models:
            raise ValueError(f'Model {setup['model']} not found. Available models: {list(available_models.keys())}.')

        setup['model'] = available_models[setup['model']]

    if isinstance(setup['loss_function'], str):
        loss_name = setup['loss_function'].split('.')[-1]

        if not hasattr(torch.nn, loss_name):
            raise ValueError(f'Loss function {setup['loss_function']} not found in torch.nn.')

        setup['loss_function'] = getattr(torch.nn, loss_name)

    if isinstance(setup['optimizer'], str):
        setup['optimizer'] = {'name': setup['optimizer'],
                              'params': {}}

    if not isinstance(setup['optimizer'], dict):
        raise ValueError('Optimizer configuration must be a string or dictionary.')

    if 'name' not in setup['optimizer']:
        raise ValueError('Optimizer configuration must contain "name".')

    optimizer_name = setup['optimizer']['name']
    optimizer_class_name = optimizer_name.split('.')[-1]

    if not hasattr(torch.optim, optimizer_class_name):
        raise ValueError(f'Optimizer: {optimizer_name} not found in torch.optim.')

    setup['optimizer']['class'] = getattr(torch.optim, optimizer_class_name)
    setup['optimizer'].setdefault('params', {})

    if isinstance(setup['scheduler'], str):
        setup['scheduler'] = {'name': setup['scheduler'],
                              'params': {}}

    if not isinstance(setup['scheduler'], dict):
        raise ValueError('Scheduler configuration must be a string or dictionary.')

    if 'name' not in setup['scheduler']:
        raise ValueError('Scheduler configuration must contain "name".')

    scheduler_name = setup['scheduler']['name']
    scheduler_class_name = scheduler_name.split('.')[-1]

    if not hasattr(torch.optim.lr_scheduler, scheduler_class_name):
        raise ValueError(f'Scheduler: {scheduler_name} not found in torch.optim.lr_scheduler.')

    setup['scheduler']['class'] = getattr(torch.optim.lr_scheduler, scheduler_class_name)
    setup['scheduler'].setdefault('params', {})

    # Optional weights for ResNetModel
    if setup['model'] is ResNetModel:
        if 'weights_ver' in setup:
            if not isinstance(setup['weights_ver'], int):
                raise ValueError('"weights_ver" must be an integer.')

    return setup

def validate_prep(prep: dict[str, Any], required_keys: list[str]) -> dict[str, Any]:
    """
    Checks for the missing content of the preprocessing configuration. 
    Casts into objects.

    Args:
        prep (dict[str, Any]): Dictionary containing the preprocessing configuration variables.
        required_keys [list[str]]: List of keys that are required in the configuration file.

    Returns:
        dict[str, Any]: Dictionary containing the preprocessing configuration variables.
            String are casted into objects.

    Raises:
        ValueError: If any of the obligatory preprocessing configuration variables are missing.
    """

    missing_keys = [key for key in required_keys if key not in prep]

    if missing_keys:
        raise ValueError(f'Missing obligatory preprocessing variables: {missing_keys}')

    interpolation_map = {'NEAREST': v2.InterpolationMode.NEAREST,
                         'NEAREST_EXACT': v2.InterpolationMode.NEAREST_EXACT,
                         'BILINEAR': v2.InterpolationMode.BILINEAR,
                         'BICUBIC': v2.InterpolationMode.BICUBIC}
    interpolation = prep['interpolation_type']

    if isinstance(interpolation, str):
        try:
            prep['interpolation_type'] = interpolation_map[interpolation.upper()]
        except KeyError:
            raise ValueError(f'Unknown interpolation type: {interpolation}. Available: {list(interpolation_map)}.')

    if len(prep['mean']) != len(prep['std']):
        raise ValueError('"mean" and "std" must have the same length.')

    if any(std <= 0 for std in prep['std']):
        raise ValueError('All values in "std" must be greater than zero.')

    if prep['resize_size'] <= 0:
        raise ValueError('"resize_size" must be positive.')

    if prep['center_crop'] <= 0:
        raise ValueError('"center_crop" must be positive.')

    if prep['center_crop'] > prep['resize_size']:
        raise ValueError('"center_crop" cannot be larger than "resize_size".')

    return prep

def validate_aug(aug: dict[str, Any], required_keys: list[str]) -> dict[str, Any]:
    """
    Checks for the missing content of the augmentation configuration. 
    Casts into objects.

    Args:
        aug (dict[str, Any]): Dictionary containing the augmentation configuration variables.
        required_keys [list[str]]: List of keys that are required in the configuration file.

    Returns:
        dict[str, Any]: Dictionary containing the augmentation configuration variables.
            String are casted into objects.

    Raises:
        ValueError: If any of the obligatory augmentation configuration variables are missing.
    """

    missing_keys = [key for key in required_keys if key not in aug]

    if missing_keys:
        raise ValueError(f'Missing obligatory augmentation variables: {missing_keys}')

    if not aug:
        return aug

    # TO DO: AUGMENTATIONS

    return aug

def build_model(setup: dict[str, Any], classification: bool) -> ClassificationModel | SegmentationModel:
    """
    Builds classification or segmentation model instance.

    Args:
        setup (dict[str, Any]): Dictionary containing configuration of the setup.
        classification (bool): Selection of classification or segmentation models.

    Returns:
        ClassificationModel | SegmentationModel: Instance of classification or segmentation model.
    """

    model_class = setup['model']

    if classification:
        model_kwargs = {'in_channels': setup['in_channels'],
                        'num_classes': setup['num_classes'],
                        'm_type': setup['m_type'],
                        'pretrained': setup['pretrained']}
    else:
        model_kwargs = {'in_channels': setup['in_channels'],
                        'encoder': setup['encoder'],
                        'pretrained': setup['pretrained'],
                        'weights_ver': setup['weights_ver']}

    if model_class is ResNetModel and 'weights_ver' in setup:
        model_kwargs['weights_ver'] = setup['weights_ver']

    return model_class(**model_kwargs)

def build_datasets(setup: dict[str, Any], prep: dict[str, Any], aug: dict[str, Any], seed: int, classification: bool) -> tuple[DatasetBase, DatasetBase]:
    """
    Splits data and converts into training and evaluation datasets.

    Args:
        setup (dict[str, Any]): Dictionary containing configuration of the setup.
        prep (dict[str, Any]): Dictionary containing configuration of the preprocessing.
        aug (dict[str, Any]): Dictionary containing configuration of the augmentations.
        seed (int): Seed.
        classification (bool): Selection of classification or segmentation datasets.

    Returns:
        tuple[DatasetBase, DatasetBase]: Train and eval datasets.
    """

    datasplit = DataSplitClassification if classification else DataSplitSegmentation

    folds = datasplit(setup['data_dir'], setup['n_folds'], seed)
    used_fold = setup['used_fold']
    rgb = setup['in_channels'] == 3

    dataset = DatasetClassification if classification else DatasetSegmentation
    train_dataset = dataset(dataset_name = setup['data_dir'], data = folds.folds[used_fold]['train'], prep = prep, aug = aug, train = True, rgb = rgb)
    val_dataset = dataset(dataset_name = setup['data_dir'], data = folds.folds[used_fold]['val'], prep = prep, train = False, rgb = rgb)
    return train_dataset, val_dataset

def build_loss_function(setup: dict[str, Any]) -> torch.nn:
    """
    Converts string into torch.nn loss function object.

    Args:
        setup (dict[str, Any]): Dictionary containing configuration of the setup.
    
    Returns:
        torch.nn: Loss function object.
    """

    loss_class = setup['loss_function']
    return loss_class()

def build_optimizer(setup: dict[str, Any], parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """
    Converts string into torch.optim optimizer object.

    Args:
        setup (dict[str, Any]): Dictionary containing configuration of the setup.
        parameters (Iterator[torch.nn.Parameter]): Parameters used in the model.
    
    Returns:
        torch.optim: Optimizer object.
    """
    
    optimizer_config = setup['optimizer']
    optimizer_class = optimizer_config['class']
    optimizer_params = optimizer_config.get('params', {})
    return optimizer_class(parameters, **optimizer_params)

def build_scheduler(setup: dict[str, Any], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
    """
    Converts string into torch.optim.lr_scheduler scheduler object.

    Args:
        setup (dict[str, Any]): Dictionary containing configuration of the setup.
        optimizer (torch.optim.Optimizer): Optimizer.
    
    Returns:
        torch.optim.lr_scheduler: Scheduler object.
    """
    
    scheduler_config = setup['scheduler']
    scheduler_class = scheduler_config['class']
    scheduler_params = scheduler_config.get('params', {})
    return scheduler_class(optimizer, **scheduler_params)