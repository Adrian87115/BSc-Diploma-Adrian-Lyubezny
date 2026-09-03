from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

class Logger:
    """
    Class responsible for managing saved models and results.

    Args:
        experiment_name (str): 
        setup (dict[str, Any]): Dictionary containing setup parameters.
        prep (dict[str, Any]): Dictionary containing preprocessing parameters.
        aug (dict[str, Any]): Dictionary containing augmentation parameters.
        run_index (int | None, optional): Selecting existing run. Defaults to None.

    Raises:
        ValueError: When attempting to select existing run without loading the model.
    """

    def __init__(self, experiment_name: str, setup: dict[str, Any], prep: dict[str, Any], aug: dict[str, Any], run_index: int | None = None):
        self.experiment_name = experiment_name
        self.setup = setup
        self.prep = prep
        self.aug = aug

        self.experiment_dir = Path(__file__).resolve().parents[1] / 'models' / experiment_name
        self.experiment_dir.mkdir(parents = True, exist_ok = True)

        self.is_resuming = run_index is not None
        self.is_loaded = False

        if run_index is None:
            self.run_index = self._get_next_run_index()
        else:
            self.run_index = run_index

        self.run_dir = self.experiment_dir / f'run_{self.run_index:03d}'

        if self.is_resuming and not self.run_dir.exists():
            raise ValueError(f'Run {self.run_index:03d} does not exist: {self.run_dir}.')

        self.run_dir.mkdir(parents = True, exist_ok = True)

        self.log_file = self.run_dir / 'log.txt'
        self.loss_file = self.run_dir / 'losses.csv'

        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents = True, exist_ok = True)

    def mark_loaded(self) -> None:
        """
        Marks the current run as safely loaded. Called by TrainBase.load_model().
        Prevents overwritting the run at selected index.
        """

        self.is_loaded = True

    def save_info(self, load: bool = False, changed_db: str = None) -> None:
        """
        Saves all necessary information about the current run.
        
        When the run is new, it saves all information. Only new used
        dataset is added.

        Args:
            load (bool, optional): If loaded, all info is being saved.
                Defaults to False.
            changed_db (str, optional): Name of new dataset. Defaults to None.
        """

        if self.is_resuming and not load:
            raise RuntimeError(f'Run {self.run_index:03d} was selected for resumption, but load = True was not specified.')

        if load:
            with open(self.log_file, 'a') as f:
                f.write('\n----------- Resumed training -----------\n')

                if changed_db:
                    f.write(f'Using {changed_db} dataset.\n')

            return

        with open(self.log_file, 'w') as f:
            f.write(f'Experiment: {self.experiment_name}\n')
            f.write(f'Run index: {self.run_index}\n')
            f.write(f'Run directory: {self.run_dir}\n')
            f.write(f'Setup: {self.setup}\n')
            f.write(f'Prep: {self.prep}\n')
            f.write(f'Aug: {self.aug}\n')
            f.write('\n')

    def _get_next_run_index(self) -> int:
        """
        Obtains index for the lastest run, and sets new for current.

        Returns:
            int: New index number.
        """

        indices = []

        for path in self.experiment_dir.glob('run_*'):
            if not path.is_dir():
                continue

            match = re.fullmatch(r'run_(\d+)', path.name)

            if match:
                indices.append(int(match.group(1)))

        if not indices:
            return 1

        return max(indices) + 1

    def update_log(self, data: str) -> None:
        """
        Updates the log of the current run, using provided data.

        Args:
            data (str): Data to be saved on the file.
        """

        with open(self.log_file, 'a') as f:
            f.write(f'{data}\n')

    def get_checkpoint_path(self, epoch: int) -> Path:
        """
        Obtains path to the selected saved model at the given epoch.

        Args:
            epoch (int): Epoch from which the training resumes.

        Returns:
            Path: Path to the selected saved model.
        """

        return (self.checkpoint_dir / f'epoch_{epoch:03d}.pt')

    def save_losses(self, epoch: int, train_loss: float, val_loss: float = None) -> None:
        """
        Saves losses of the current epoch (epoch, train_loss, val_loss) in the .csv file.

        In the case of resumption, it will overwrite existing epochs.
        
        Args:
            epoch (int): Current epoch.
            train_loss (float): Training loss at the given epoch.
            val_loss (float | None, optional): Evaluation loss at the given epoch.
                Defaults to None.
        """

        if not self.loss_file.exists():
            with open(self.loss_file, 'w') as f:
                f.write('epoch,train_loss,val_loss\n')

        rows = {}

        with open(self.loss_file, 'r') as f:
            next(f, None)

            for line in f:
                line = line.strip()

                if not line:
                    continue

                values = line.split(',')

                saved_epoch = int(values[0])
                rows[saved_epoch] = values

        rows[epoch] = [str(epoch), str(train_loss), '' if val_loss is None else str(val_loss)]

        with open(self.loss_file, 'w') as f:
            f.write('epoch,train_loss,val_loss\n')

            for saved_epoch in sorted(rows):
                f.write(','.join(rows[saved_epoch]) + '\n')

    def plot_losses(self, save: bool = True, show: bool = False) -> None:
        """
        Plot training and validation losses from losses.csv.

        Args:
            save (bool, optional): Save the plot as losses.png.
                Defaults to True.
            show (bool, optional): Display the plot interactively.
                Defaults to False.
        """

        if not self.loss_file.exists():
            raise ValueError(f'Loss file does not exist: {self.loss_file}')

        data = pd.read_csv(self.loss_file)

        if data.empty:
            raise ValueError(f'Loss file is empty: {self.loss_file}')

        plt.figure()
        plt.plot(data['epoch'], data['train_loss'], label = 'Train Loss')

        if data['val_loss'].notna().any():
            val_data = data.dropna(subset = ['val_loss'])
            plt.plot(val_data['epoch'], val_data['val_loss'], label = 'Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.experiment_name} - Run {self.run_index:03d}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            plot_file = self.run_dir / 'losses.png'
            plt.savefig(plot_file, dpi = 300)

        if show:
            plt.show()

        plt.close()