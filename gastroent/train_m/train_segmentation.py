import torch
import torch.distributed as dist
import time
from typing import Any

from train_m.train_base import TrainBase
from data_m.transforms import get_transforms

class TrainSegmentation(TrainBase):
    def evaluate(self, prep: dict[str, Any], epoch: int = None, gpu_transforms: bool = True) -> tuple[float, str]:
        """
        Evaluates the segmentation model on the validation dataset.

        Computes average loss, Dice Coefficient, and Intersection over Union (IoU).

        Args:
            prep: (dict[str, Any]): Dictionary containing preprocessing parameters.
            epoch (int | None, optional): Current epoch number for logging. 
                Defaults to None (uses self.resumed_epoch).
            gpu_transforms (bool, optional): Using GPU to transforming images. Defaults to True.

        Returns:
            tuple[float, str]: A tuple containing the average loss and the 
                formatted evaluation statistics string.
        """

        transforms = get_transforms(prep = prep, aug = None, use_gpu = gpu_transforms, classification = False)

        if not epoch:
            epoch = self.resumed_epoch

        start_time = time.time()
        self.ddp.eval()

        total_loss = torch.tensor(0.0, device = self.device)
        total_dice = torch.tensor(0.0, device = self.device)
        total_iou = torch.tensor(0.0, device = self.device)
        total_samples = torch.tensor(0, device = self.device)

        if self.rank == 0:
            print('Evaluation Started...')

        with torch.no_grad():
            for inputs, masks in self.eval_dataloader:
                if gpu_transforms:
                    inputs = inputs.to(self.device, non_blocking = True)
                    masks = masks.to(self.device, non_blocking = True)
                    inputs, masks = transforms(inputs, masks)
                else:
                    inputs_np = inputs.numpy()
                    masks_np = masks.numpy()
                    
                    transformed_inputs = []
                    transformed_masks = []

                    for image, mask in zip(inputs_np, masks_np):
                        image = image.transpose(1, 2, 0)

                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask.squeeze(0)

                        transformed = transforms(image = image, mask = mask)
                        transformed_inputs.append(transformed['image'])
                        transformed_masks.append(transformed['mask'])

                    inputs = torch.stack(transformed_inputs).to(self.device, non_blocking = True)
                    masks = torch.stack(transformed_masks).to(self.device, non_blocking = True)

                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)

                outputs = self.ddp(inputs)
                loss = self.loss_function(outputs, masks)

                batch_size = masks.size(0)
                total_loss += loss * batch_size
                total_samples += batch_size

                preds = torch.argmax(outputs, dim = 1).float()
                masks = masks.float()
                preds_flat = preds.view(batch_size, -1)
                masks_flat = masks.view(batch_size, -1)

                intersection = (preds_flat * masks_flat).sum(dim = 1)
                union = preds_flat.sum(dim = 1) + masks_flat.sum(dim = 1) - intersection
                
                dice = (2.0 * intersection) / (preds_flat.sum(dim = 1) + masks_flat.sum(dim = 1) + 1e-8)
                iou = intersection / (union + 1e-8)

                total_dice += dice.sum()
                total_iou += iou.sum()

        if self.world_size > 1:
            dist.all_reduce(total_loss, op = dist.ReduceOp.SUM)
            dist.all_reduce(total_dice, op = dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op = dist.ReduceOp.SUM)
            dist.all_reduce(total_iou, op = dist.ReduceOp.SUM)

        avg_loss = (total_loss / total_samples).item()
        avg_dice = (total_dice / total_samples).item()
        avg_iou = (total_iou / total_samples).item()

        eval_stat = (f'Eval Results (Epoch {epoch}) | Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f} | Time: {time.time() - start_time:.2f}s')

        if self.rank == 0:
            print(eval_stat)
            print('Evaluation Completed.')

        return avg_loss, eval_stat