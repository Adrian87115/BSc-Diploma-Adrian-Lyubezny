from pathlib import Path
import os

from train_m.train_classification import TrainClassification
from train_m.train_segmentation import TrainSegmentation
from utils_m.logger import Logger
from utils_m.utils import (get_config, build_model, build_datasets,
                           build_loss_function, build_optimizer, build_scheduler)

def build_experiment(seed: int, experiment_name: str, run_index: int | None = None, epoch_load: int | None = None, train: bool = True, plot_losses: bool = False, classification: bool = True) -> None:
    """
    Args:
        seed (int): Seed to provide reproducible results.
        experiment_name (str): Name of the experiment. All runs are stored under this name.
        run_index (int | None, optional): Loading the specified run. If not present, new one is created.
            Defaults to None.
        epoch_load (int | None, optional): Loading the model from the given epoch number. Defaults to None.
        train (bool, optional): Training or evaluation modes. Defaults to True.
        plot_losses (bool, optional): Plotting the saved losses - only if present. Defaults to False.
        classification (bool, optional): Selection between classification and segmentation functionalities.
            Defaults to True.
    """

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))

    if run_index is not None:
        config_path = Path(__file__).resolve().parents[2] / 'models' / experiment_name / f'run_{run_index:03d}' / 'launch_config.json'

        if not config_path.exists():
            raise FileNotFoundError(f'Config file for resumption not found: {config_path}.')
    else:
        config_path = Path(__file__).resolve().parents[2] / 'models' / experiment_name / 'launch_config.json'

        if not config_path.exists():
            raise FileNotFoundError(f'Base config file not found: {config_path}.')

    setup, prep, aug = get_config(config_path, classification = classification)

    logger = Logger(experiment_name = experiment_name, setup = setup, prep = prep, aug = aug, run_index = run_index)

    if run_index is None and rank == 0:
        logger.save_config(config_path)

    if plot_losses and run_index is not None:
        logger.plot_losses()

    model = build_model(setup = setup, classification = classification)

    train_dataset, val_dataset = build_datasets(setup = setup, prep = prep, seed = seed, classification = classification)

    loss_function = build_loss_function(setup = setup)

    optimizer = build_optimizer(setup = setup, parameters = model.parameters())

    scheduler = build_scheduler(setup = setup, optimizer = optimizer)

    TrainerClass = TrainClassification if classification else TrainSegmentation
                         
    trainer = TrainerClass(logger = logger, model = model, batch_size = setup['batch_size'],
                           loss_function = loss_function,
                           train_dataset = train_dataset, eval_dataset = val_dataset,
                           optimizer = optimizer, scheduler = scheduler,
                           num_workers = setup['num_workers'], persistent_workers = setup['persistent_workers'],
                           prefetch_factor = setup['prefetch_factor'], pin_memory = setup['pin_memory'],
                           seed = seed, backend_mode = setup['backend_mode'], rank = rank, world_size = world_size, 
                           d_sampler_params = setup['d_sampler_params'])

    if run_index is not None and epoch_load is not None:
        trainer.load_model(epoch_load)

    if train:
        trainer.train(num_epochs = setup['num_epochs'], prep = prep, aug = aug, save_freq = setup['save_freq'], eval_freq = setup['eval_freq'], gpu_transforms = setup['gpu_transforms'])
    else:
        if classification:
            trainer.evaluate(prep = prep, detailed = True, unique_labels = val_dataset.get_unique_labels(), gpu_transforms = setup['gpu_transforms'])
        else:
            trainer.evaluate(prep = prep, gpu_transforms = setup['gpu_transforms'])

def classification_experiment(seed: int, experiment_name: str, run_index: int | None = None, epoch_load: int | None = None, train: bool = True, plot_losses: bool = False) -> None:
    """
    Runs the classification experiment.
    """

    print(f'Running classification experiment: {experiment_name}.')
    build_experiment(seed, experiment_name, run_index, epoch_load, train, plot_losses, classification = True)
    print('Classification experiment complete.')

def segmentation_experiment(seed: int, experiment_name: str, run_index: int | None = None, epoch_load: int | None = None, train: bool = True, plot_losses: bool = False) -> None:
    """
    Runs the segmentation experiment.
    """
    
    print(f'Running segmentation experiment: {experiment_name}.')
    build_experiment(seed, experiment_name, run_index, epoch_load, train, plot_losses, classification = False)
    print('Segmentation experiment complete.')