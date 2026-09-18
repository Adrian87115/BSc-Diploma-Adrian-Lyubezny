import torch
from torch.utils.data import DataLoader, DistributedSampler
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any

from utils_m.utils import set_seeds, setup, cleanup
from data_m.transforms import get_transforms
from utils_m.logger import Logger
from models_m.classification_models import ClassificationModel
from models_m.segmentation_models import SegmentationModel
from data_m.dataset_base import DatasetBase

class TrainBase():
    """
    Base for classification and segmentation, with common training loop.

    Args:
        logger (Logger): Object used to handle: logging, saving, and loading.
        model (ClassificationModel | SegmentationModel): Wrapper object with ready to use self.model.
        batch_size (int): Size of the batch used in the training and evaluation. 
        loss_function (torch.nn): Loss function.
        train_dataset (DatasetBase): Training data. 
        eval_dataset (DatasetBase): Evaluation data.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        scheduler (torch.optim.lr_scheduler): Scheduling algorithm.
        num_workers (int): Number of CPU workers for data loading.
        persistent_workers (bool): If True, keeps the data loader worker processes alive between epochs.
        prefetch_factor (int): Number of batches loaded in advance by each worker to prevent GPU starvation.
        pin_memory (bool): If True, allocates data in page-locked memory, which speeds up the data transfer from CPU to GPU.
        seed (int): Seed used to reproduce the results.
        backend_mode (str): Backend for distributed training ('gloo' for Windows, 'nccl' for Linux).
        rank (int, optional): The index of the current process in the distributed training setup (0 is the main/master process).
            Defaults to 0.
        world_size (int, optional): The total number of processes/GPUs participating in the distributed training. Defaults to 1.
    """

    def __init__(self, logger: Logger, model: ClassificationModel | SegmentationModel, 
                 batch_size: int, loss_function: torch.nn, train_dataset: DatasetBase, 
                 eval_dataset: DatasetBase, optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler, num_workers: int, 
                 persistent_workers: bool, prefetch_factor: int, pin_memory: bool, 
                 seed: int = 42, backend_mode: str = 'gloo', rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size

        if self.world_size > 1: 
            setup(self.world_size, self.rank, backend_mode)

        self.device = torch.device(f'cuda:{self.rank}')
        print(f'Training on device: {self.rank}')

        self.logger = logger
        self.model = model.model.to(self.device)
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.seed = seed

        self.num_classes = getattr(model, 'num_classes', 2)

        if self.world_size > 1:
            self.ddp = DDP(self.model, device_ids = [self.rank], output_device = self.rank)
        else:
            self.ddp = self.model

        self.resumed_epoch = 0

        self.experiment_dir = self.logger.experiment_dir
        self.run_dir = self.logger.run_dir
        self.run_index = self.logger.run_index

        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents = True, exist_ok = True)

        set_seeds(self.seed + self.rank)

        if self.world_size > 1: 
            self.train_sampler = DistributedSampler(train_dataset, num_replicas = self.world_size, rank = self.rank, shuffle = True) 
            self.val_sampler = DistributedSampler(eval_dataset, num_replicas = self.world_size, rank = self.rank, shuffle = False) 
        else: 
            self.train_sampler = None 
            self.val_sampler = None 

        shuffle = False if self.train_sampler is not None else True
        self.train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, sampler = self.train_sampler, shuffle = shuffle, num_workers = num_workers, persistent_workers = persistent_workers, prefetch_factor = prefetch_factor, pin_memory = pin_memory) 
        self.eval_dataloader = DataLoader(eval_dataset, batch_size = self.batch_size, sampler = self.val_sampler, shuffle = False, num_workers = num_workers, persistent_workers = persistent_workers, prefetch_factor = prefetch_factor, pin_memory = pin_memory) 

        dataset_name = getattr(train_dataset, 'dataset_name', 'Unknown')
        self.used_datasets = [dataset_name] if isinstance(dataset_name, str) else list(dataset_name)

    def cleanup(self) -> None:
        """
        Cleans up and destroys the distributed training environment.
        """
         
        if self.world_size > 1: 
            cleanup()

    def save_model(self, epoch: int) -> None:
        """
        Saves the model state.

        Args:
            epoch (int): Epoch number used to name the file, and
                to resume the training from this point.
        """

        model_name = self.model.__class__.__name__
        checkpoint = {'run_index': self.run_index,
                      'model_name': model_name,
                      'epoch': epoch,
                      'datasets': self.used_datasets,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}

        file_path = self.logger.get_checkpoint_path(epoch)
        torch.save(checkpoint, file_path)

        print(f'Model successfully saved to {file_path}.')

    def load_model(self, epoch: int) -> None:
        """
        Load the model state.

        Args:
            epoch (int): Number of epoch, used to load the state.

        Raises:
            ValueError: If checkpoint was not found.
                        If used and saved architectures mismatch.
                        If run indices do not match.
        """

        file_path = self.logger.get_checkpoint_path(epoch)

        if not file_path.exists():
            raise ValueError(f'Checkpoint for epoch {epoch} was not found: {file_path}.')

        checkpoint = torch.load(file_path, map_location = self.device)

        current_model_name = self.model.__class__.__name__
        saved_name = checkpoint.get('model_name', 'Unknown')

        if current_model_name != saved_name:
            raise ValueError(f'Model architecture mismatch. Saved: {saved_name}, currently used: {current_model_name}.')

        saved_run_index = checkpoint.get('run_index')

        if saved_run_index != self.run_index:
            raise ValueError(f'Run mismatch. Logger: {self.run_index}, checkpoint: {saved_run_index}.')

        self.resumed_epoch = checkpoint['epoch']
        
        saved_datasets = checkpoint.get('datasets', [])

        for dataset in self.used_datasets:
            if dataset not in saved_datasets:
                saved_datasets.append(dataset)

        self.used_datasets = saved_datasets

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_state = checkpoint.get('scheduler_state_dict')

        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.logger.mark_loaded()

        print(f'Model successfully loaded from: {file_path}. Resuming epoch: {self.resumed_epoch}.')

    def train(self, num_epochs: int, prep: dict[str, Any], aug: dict[str, Any] | None = None, save_freq: int = 5, eval_freq: int = 1, print_batch: bool = True, gpu_transforms: bool = True) -> None:
        """
        Train, validate, and save the model.

        Args:
            num_epochs (int): Number of training epochs.
            prep (dict[str, Any]): Dicitonary with parameters for preprocessing the images.
            aug (dict[str, Any] | None, optional): Dicitonary with parameters for augmentation of the images.
            save_freq (int, optional): Frequency of saving the state of the model. Defaults to 5.
            eval_freq (int, optional): Frequency of evaluation. Defaults to 1.
            print_batch (bool, optional): Enables printing statistics of each batch. Defaults to True.
            gpu_transforms (bool, optional): Preprocessing and augmentations on GPU. Defaults to True.
        
        Raises:
            RuntimeError: If training was resumed in the selected run, but model was not loaded.
        """

        from train_m.train_classification import TrainClassification
        classification = isinstance(self, TrainClassification)
        
        transforms = get_transforms(prep = prep, aug = aug, use_gpu = gpu_transforms, classification = classification)

        if self.logger.is_resuming and not self.logger.is_loaded:
            raise RuntimeError(f'trainer.load_model() must be called before trainer.train() while resuming.')
    
        start_epoch = self.resumed_epoch + 1
        self.logger.save_info(load = start_epoch != 1)
        
        print('Training Started...')

        for epoch in range(start_epoch, num_epochs + 1):
            t_start_epoch = time.time()

            self.ddp.train()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch) 

            total_loss = 0.0
            total_samples = 0
            
            for i, (inputs, labels) in enumerate(self.train_dataloader):
                t_start_batch = time.time()

                if gpu_transforms:
                    inputs = inputs.to(self.device, non_blocking = True)
                    labels = labels.to(self.device, non_blocking = True)

                    if classification:
                        inputs = transforms(inputs)
                    else:
                        inputs, labels = transforms(inputs, labels)

                        if labels.ndim == 4 and labels.shape[1] == 1:
                            labels = labels.squeeze(1)
                else:
                    inputs_np = inputs.numpy()
                    transformed_inputs = []

                    if classification:
                        for image in inputs_np:
                            image = image.transpose(1, 2, 0)
                            transformed = transforms(image = image,)
                            transformed_inputs.append(transformed['image'])
                    else:
                        labels_np = labels.numpy()
                        transformed_labels = []

                        for image, mask in zip(inputs_np, labels_np):
                            image = image.transpose(1, 2, 0)

                            if mask.ndim == 3 and mask.shape[0] == 1:
                                mask = mask.squeeze(0)

                            transformed = transforms(image = image, mask = mask)
                            transformed_inputs.append(transformed['image'])
                            transformed_labels.append(transformed['mask'])

                    inputs = torch.stack(transformed_inputs).to(self.device, non_blocking = True)
                    labels = torch.stack(transformed_labels).to(self.device, non_blocking = True) if not classification else labels.to(self.device, non_blocking = True)

                    if not classification and labels.ndim == 4 and labels.shape[1] == 1:
                        labels = labels.squeeze(1)

                self.optimizer.zero_grad(set_to_none = True)
            
                outputs = self.ddp(inputs)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

                batch_loss = loss.item()
                current_batch_size = inputs.size(0)
                total_loss += batch_loss * current_batch_size
                total_samples += current_batch_size

                batch_time_ms = (time.time() - t_start_batch) * 1000

                if print_batch:
                    print(f'Epoch [{epoch} / {num_epochs}] | Batch [{i + 1} / {len(self.train_dataloader)}] | Loss: {batch_loss:.4f} | Time: {batch_time_ms:.2f}ms')

            current_lr = self.optimizer.param_groups[0]['lr']
            avg_train_loss = total_loss / total_samples
            epoch_stat = f'Epoch {epoch} | Avg Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | Time: {time.time() - t_start_epoch:.2f}s'
            print(epoch_stat)

            if self.rank == 0:
                self.logger.update_log(epoch_stat)

            eval_loss = None

            if epoch % eval_freq == 0:
                eval_loss, eval_stat = self.evaluate(epoch = epoch, prep = prep, gpu_transforms = gpu_transforms)

                if self.rank == 0:
                    self.logger.update_log(eval_stat)

            if epoch % save_freq == 0 and self.rank == 0:
                self.save_model(epoch)

            if self.rank == 0:
                self.logger.save_losses(epoch, avg_train_loss, eval_loss)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    pass
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if eval_loss is not None:
                        self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()

        self.cleanup()
        print('Training Completed.')