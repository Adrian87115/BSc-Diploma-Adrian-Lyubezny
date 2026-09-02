import torch
import time

from train_m.train_base import TrainBase

class TrainSegmentation(TrainBase):
    def evaluate(self, epoch: int = None) -> tuple[float, str]:
        """
        Evaluates the segmentation model on the validation dataset.

        Computes average loss, Dice Coefficient, and Intersection over Union (IoU).

        Args:
            epoch (int | None, optional): Current epoch number for logging. 
                Defaults to None (uses self.resumed_epoch).

        Returns:
            tuple[float, str]: A tuple containing the average loss and the 
                formatted evaluation statistics string.
        """

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
                inputs = inputs.to(self.device, non_blocking = True)
                masks = masks.to(self.device, non_blocking = True)

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
            torch.distributed.all_reduce(total_loss, op = torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_dice, op = torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_samples, op = torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_iou, op = torch.distributed.ReduceOp.SUM)

        avg_loss = (total_loss / total_samples).item()
        avg_dice = (total_dice / total_samples).item()
        avg_iou = (total_iou / total_samples).item()

        eval_stat = (f'Eval Results (Epoch {epoch}) | Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f} | Time: {time.time() - start_time:.2f}s')

        if self.rank == 0:
            print(eval_stat)
            print('Evaluation Completed.')

        return avg_loss, eval_stat