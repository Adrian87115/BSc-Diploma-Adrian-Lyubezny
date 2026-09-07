from pathlib import Path
import os

from train_m.train_classification import TrainClassification
from train_m.train_segmentation import TrainSegmentation
from utils_m.logger import Logger
from utils_m.utils import (get_config, build_model, build_datasets,
                           build_loss_function, build_optimizer, build_scheduler)

def build_experiment(seed, experiment_name, run_index = None, epoch_load = None, train = True, plot_losses = False, classification = True):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))

    config_path = Path(__file__).resolve().parents[2] / 'models' / experiment_name / 'launch_config.json'
    setup, prep, aug = get_config(config_path, classification = classification)

    log = Logger(experiment_name = experiment_name, setup = setup, prep = prep, aug = aug, run_index = run_index)

    if plot_losses and run_index is not None:
        log.plot_losses()

    model = build_model(setup = setup, classification = classification)

    train_dataset, val_dataset = build_datasets(setup = setup, prep = prep, aug = aug, seed = seed, classification = classification)

    loss_function = build_loss_function(setup = setup)

    optimizer = build_optimizer(setup = setup, parameters = model.parameters())

    scheduler = build_scheduler(setup = setup, optimizer = optimizer)

    TrainerClass = TrainClassification if classification else TrainSegmentation

    trainer = TrainerClass(logger = log, model = model, batch_size = setup['batch_size'],
                           loss_function = loss_function,
                           train_dataset = train_dataset, eval_dataset = val_dataset,
                           optimizer = optimizer, scheduler = scheduler,
                           num_workers = setup['num_workers'], persistent_workers = setup['persistent_workers'],
                           prefetch_factor = setup['prefetch_factor'], pin_memory = setup['pin_memory'],
                           seed = seed, rank = rank, world_size = world_size)

    if run_index is not None and epoch_load is not None:
        trainer.load_model(epoch_load)

    if train:
        trainer.train(num_epochs = setup['num_epochs'], save_freq = setup['save_freq'], eval_freq = setup['eval_freq'])
    else:
        if classification:
            trainer.evaluate(detailed = True, unique_labels = val_dataset.get_unique_labels())
        else:
            trainer.evaluate()

def classification_experiment(seed, experiment_name, run_index = None, epoch_load = None, train = True, plot_losses = False):
    print(f'Running classification experiment: {experiment_name}.')
    build_experiment(seed, experiment_name, run_index, epoch_load, train, plot_losses, classification = True)
    print('Classification experiment complete.')

def segmentation_experiment(seed, experiment_name, run_index = None, epoch_load = None, train = True, plot_losses = False):
    print(f'Running segmentation experiment: {experiment_name}.')
    build_experiment(seed, experiment_name, run_index, epoch_load, train, plot_losses, classification = False)
    print('Segmentation experiment complete.')