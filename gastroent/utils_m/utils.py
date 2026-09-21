import random
import numpy as np
import torch
import torch.distributed as dist
import os
import json
from pathlib import Path
from typing import Any
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
                                 'num_epochs', 'save_freq', 'eval_freq', 'backend_mode',
                                 'd_sampler_params']

# Required keys for building and running segmentation tasks
REQUIRED_SETUP_SEGMENTATION = ['in_channels', 'encoder', 'pretrained',
                               'weights_ver', 'num_workers', 'persistent_workers',
                               'prefetch_factor', 'pin_memory', 'model',
                               'data_dir', 'n_folds', 'used_fold',
                               'loss_function', 'optimizer', 'scheduler', 'batch_size',
                               'num_epochs', 'save_freq', 'eval_freq', 'backend_mode']

# Required keys for preprocessing
REQUIRED_PREP = ['resize_size', 'interpolation_type', 'center_crop', 'mean', 'std']

# Keys that can be used for augmentation
POSSIBLE_AUG = ['hsv', 'rotation', 'translation', 'scale', 'interpolation_type', 'horizontal_flip',
                'vertical_flip', 'fill', 'random_crop', 'probs']

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
    
    Raises:
        ValueError: If backend is incorrect.
    """

    if backend not in ['gloo', 'nccl']:
        raise ValueError(f'Incorrect backend: {backend}. Allowed: gloo, nccl.')
    
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

def to_hw(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
        """
        Expands 1 dimensional size to 2 dimensional height x width, if needed.

        Args:
            size (int | tuple[int, int] | list[int]): 1 or 2 dimensional size.

        Returns:
            tuple[int, int]: New size.
        """

        if isinstance(size, int):
            return size, size

        if len(size) != 2:
            raise ValueError(f'Expected an integer or a 2-element size, got {size}.')

        return int(size[0]), int(size[1])

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
            raise ValueError(f"Model {setup['model']} not found. Available models: {list(available_models.keys())}.")

        setup['model'] = available_models[setup['model']]

    if isinstance(setup['loss_function'], str):
        loss_name = setup['loss_function'].split('.')[-1]

        if not hasattr(torch.nn, loss_name):
            raise ValueError(f"Loss function {setup['loss_function']} not found in torch.nn.")

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
                raise ValueError("'weights_ver' must be an integer.")

    if setup['backend_mode'] not in ['gloo', 'nccl']:
         raise ValueError(f"Invalid backen mode: {setup['backend_mode']}. Expected: 'gloo', 'nccl'.")

    return setup

def validate_prep(prep: dict[str, Any], required_keys: list[str]) -> dict[str, Any]:
    """
    Checks for missing content of the preprocessing configuration. 

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

    interpolation = prep['interpolation_type']

    if interpolation not in ['bilinear', 'bicubic', 'nearest']:
        raise ValueError(f'Unknown interpolation type: {interpolation}.')         

    if len(prep['mean']) != len(prep['std']):
        raise ValueError("'mean' and 'std' must have the same length.")

    if any(std <= 0 for std in prep['std']):
        raise ValueError('All values in "std" must be greater than zero.')

    if prep['resize_size'] <= 0:
        raise ValueError("'resize_size' must be positive.")

    if prep['center_crop'] <= 0:
        raise ValueError("'center_crop' must be positive.")

    if prep['center_crop'] > prep['resize_size']:
        raise ValueError("'center_crop' cannot be larger than 'resize_size'.")

    return prep

def validate_aug(aug: dict[str, Any], allowed_keys: list[str]) -> dict[str, Any]:
    """
    Checks for invalid content of the augmentation configuration. 

    Args:
        aug (dict[str, Any]): Dictionary containing the augmentation configuration variables.
        required_keys [list[str]]: List of keys that may be used in the configuration file.

    Returns:
        dict[str, Any]: Dictionary containing the augmentation configuration variables.
            String are casted into objects.

    Raises:
        ValueError: If argument is not allowed.
                    If argument is invalid.
    """

    if not aug:
        return None
    
    invalid_keys = [key for key in aug if key not in allowed_keys]

    if invalid_keys:
        raise ValueError(f'Invalid keys in augmentation: {invalid_keys}. Available: {allowed_keys}.')

    if 'hsv' in aug and aug['hsv'] is not None:
        hsv = aug['hsv']

        if not isinstance(hsv, dict):
            raise ValueError("aug['hsv'] must be a dictionary.")
        
        for k in ['hue', 'saturation', 'brightness']:
            if k in hsv:
                val = hsv[k]

                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError(f"hsv '{k}' must be a list or tuple of 2 values (min, max).")

                if val[0] > val[1]:
                    raise ValueError(f"hsv '{k}' min value ({val[0]}) cannot be greater than max value ({val[1]}).")
                
                if k in ['saturation', 'brightness'] and val[0] < 0:
                    raise ValueError(f"hsv '{k}' values cannot be negative.")

    for key in ['rotation', 'translation', 'scale']:
        if key in aug and aug[key] is not None:
            val = aug[key]

            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"aug['{key}'] must be a list or tuple of 2 values (min, max).")

            if val[0] > val[1]:
                raise ValueError(f"aug['{key}'] min value ({val[0]}) cannot be greater than max value ({val[1]}).")
            
            if key in ['scale', 'translation'] and val[0] < 0:
                raise ValueError(f"aug['{key}'] values cannot be negative.")

    if 'interpolation_type' in aug and aug['interpolation_type'] is not None:
        valid_interpolation = ['bilinear', 'bicubic', 'nearest']

        if aug['interpolation_type'] not in valid_interpolation:
            raise ValueError(f"aug['interpolation_type'] must be one of {valid_interpolation}.")

    for key in ['horizontal_flip', 'vertical_flip']:
        if key in aug and aug[key] is not None:
            if not isinstance(aug[key], bool):
                raise ValueError(f"aug['{key}'] must be a boolean.")

    if 'fill' in aug and aug['fill'] is not None:
        fill = aug['fill']

        if not isinstance(fill, int) and fill not in ['zeros', 'border', 'reflection']:
            raise ValueError("aug['fill'] must be an integer (CPU) or one of ['zeros', 'border', 'reflection'] (GPU).")

    if 'random_crop' in aug and aug['random_crop'] is not None:
        rc = aug['random_crop']

        if isinstance(rc, int):
            if rc <= 0:
                raise ValueError(f"aug['random_crop'] must be greater than 0, got {rc}.")
        elif isinstance(rc, (list, tuple)) and len(rc) == 2:
            if rc[0] <= 0 or rc[1] <= 0:
                raise ValueError(f"aug['random_crop'] dimensions must be greater than 0, got {rc}.")
        else:
            raise ValueError("aug['random_crop'] must be an integer or a list/tuple of 2 integers.")

    if 'probs' in aug and aug['probs'] is not None:
        probs = aug['probs']

        if not isinstance(probs, dict):
            raise ValueError("aug['probs'] must be a dictionary.")
        
        valid_prob_keys = ['hsv', 'affine', 'horizontal_flip', 'vertical_flip', 'random_crop']

        for k, v in probs.items():
            if k not in valid_prob_keys:
                raise ValueError(f"Unknown probability key '{k}'. Expected one of {valid_prob_keys}.")
            
            if not isinstance(v, (float, int)) or not (0.0 <= v <= 1.0):
                raise ValueError(f"Probability for '{k}' must be a float between 0.0 and 1.0, got {v}.")

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

def build_datasets(setup: dict[str, Any], prep: dict[str, Any], seed: int, classification: bool) -> tuple[DatasetBase, DatasetBase]:
    """
    Splits data and converts into training and evaluation datasets.

    Args:
        setup (dict[str, Any]): Dictionary containing configuration of the setup.
        prep (dict[str, Any]): Dictionary containing configuration of the preprocessing.
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
    train_dataset = dataset(dataset_name = setup['data_dir'], prep = prep, data = folds.folds[used_fold]['train'], rgb = rgb)
    val_dataset = dataset(dataset_name = setup['data_dir'], prep = prep, data = folds.folds[used_fold]['val'], rgb = rgb)
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