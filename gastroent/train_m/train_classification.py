import torch
import torch.distributed as dist
import time
from torchmetrics.classification import ConfusionMatrix, AUROC, Precision, Recall, F1Score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

from train_m.train_base import TrainBase
from data_m.transforms import get_transforms

class TrainClassification(TrainBase):
    def evaluate(self, prep: dict[str, Any], epoch: int = None, detailed: bool = False, unique_labels: dict[str, int] | None = None, gpu_transforms: bool = True) -> tuple[float, str]:
        """
        Evaluates the classification model on the validation dataset.

        Computes average loss, Top-1, Top-2, and Top-3. Can optionally compute 
        and plot detailed metrics (Confusion Matrix, ROC AUC, F1, Recall, Precision).

        Args:
            prep: (dict[str, Any]): Dictionary containing preprocessing parameters.
            epoch (int | None, optional): Current epoch number for logging. 
                Defaults to None (uses self.resumed_epoch).
            detailed (bool, optional): If True, computes and plots advanced 
                classification metrics. Defaults to False.
            unique_labels (dict[str, int] | None, optional): Mapping of class 
                names to integer labels for confusion matrix axis labels. Defaults to None.
            gpu_transforms (bool, optional): Using GPU to transforming images. Defaults to True.

        Returns:
            tuple[float, str]: A tuple containing the average loss and the formatted 
                evaluation statistics string.
        """

        transforms = get_transforms(prep = prep, aug = None, use_gpu = gpu_transforms, classification = True)

        if not epoch:
            epoch = self.resumed_epoch
            
        start_time = time.time()
        self.ddp.eval()

        total_loss = torch.tensor(0.0, device = self.device)
        total_correct = torch.tensor(0, device = self.device)
        total2_correct = torch.tensor(0, device = self.device)
        total3_correct = torch.tensor(0, device = self.device)
        total_samples = torch.tensor(0, device = self.device)

        if self.rank == 0:
            print('Evaluation Started...')

        if detailed:
            confmat = ConfusionMatrix(task = 'multiclass', num_classes = self.num_classes).to(self.device)
            roc_auc = AUROC(task = 'multiclass', num_classes = self.num_classes, average = 'macro').to(self.device)
            precision = Precision(task = 'multiclass', num_classes = self.num_classes, average = 'macro').to(self.device)
            recall = Recall(task = 'multiclass', num_classes = self.num_classes, average = 'macro').to(self.device)
            f1 = F1Score(task = 'multiclass', num_classes = self.num_classes, average = 'macro').to(self.device)

        with torch.no_grad():
            for inputs, labels in self.eval_dataloader:
                if gpu_transforms:
                    inputs = inputs.to(self.device, non_blocking = True)
                    labels = labels.to(self.device, non_blocking = True)
                    inputs = transforms(inputs)
                else:
                    inputs_np = inputs.numpy()
                    transformed_inputs = []

                    for image in inputs_np:
                        image = image.transpose(1, 2, 0)
                        transformed = transforms(image = image)
                        transformed_inputs.append(transformed['image'])

                    inputs = torch.stack(transformed_inputs).to(self.device, non_blocking = True)
                    labels = labels.to(self.device,non_blocking = True)

                outputs = self.ddp(inputs)
                loss = self.loss_function(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.detach() * batch_size
                total_samples += batch_size

                preds = outputs.argmax(dim = 1)
                total_correct += (preds == labels).sum()

                if self.num_classes >= 2:
                    top2_preds = torch.topk(outputs, k = 2, dim = 1).indices
                    total2_correct += (top2_preds == labels.view(-1, 1)).sum()

                if self.num_classes >= 3:
                    top3_preds = torch.topk(outputs, k = 3, dim = 1).indices
                    total3_correct += (top3_preds == labels.view(-1, 1)).sum()

                if detailed:
                    confmat.update(preds, labels)
                    precision.update(preds, labels)
                    recall.update(preds, labels)
                    f1.update(preds, labels)
                    roc_auc.update(outputs.softmax(dim = 1), labels)
            
        if self.world_size > 1:
            dist.all_reduce(total_loss, op = dist.ReduceOp.SUM)
            dist.all_reduce(total_correct, op = dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op = dist.ReduceOp.SUM)
            dist.all_reduce(total2_correct, op = dist.ReduceOp.SUM)
            dist.all_reduce(total3_correct, op = dist.ReduceOp.SUM)

        avg_loss = (total_loss / total_samples).item()
        accuracy1 = (total_correct / total_samples).item()
        accuracy2 = (total2_correct / total_samples).item() if self.num_classes >= 2 else 0.0
        accuracy3 = (total3_correct / total_samples).item() if self.num_classes >= 3 else 0.0

        eval_stat = f'Eval Results Epoch: {epoch} | Avg Loss: {avg_loss:.4f} | Top1: {accuracy1:.4f} | Top2: {accuracy2:.4f} | Top3: {accuracy3:.4f} | Time: {time.time() - start_time:.2f}s'

        if detailed:
            cm_tensor = confmat.compute()
            roc_auc_val = roc_auc.compute().item()
            precision_val = precision.compute().item()
            recall_val = recall.compute().item()
            f1_val = f1.compute().item()

            if self.rank == 0:
                cm_numpy = cm_tensor.cpu().numpy()

                print(f'Confusion Matrix:\n {cm_tensor}\n'
                      f'ROC AUC: {roc_auc_val:.4f}\n'
                      f'Precision: {precision_val:.4f}\n'
                      f'Recall: {recall_val:.4f}\n'
                      f'F1 Score: {f1_val:.4f}')

                if unique_labels:
                    classes = [k for k, v in sorted(unique_labels.items(), key = lambda item: item[1])]
                else:
                    classes = [str(i) for i in range(1, self.num_classes + 1)]

                plt.figure(figsize = (10, 8))
                sns.heatmap(cm_numpy, annot = True, fmt = 'd', cmap = 'Blues', cbar = True, xticklabels = classes, yticklabels = classes)
                plt.title('Confusion Matrix', pad = 15)
                plt.xlabel('Predicted Label', labelpad = 10)
                plt.ylabel('True Label', labelpad = 10)
                plt.tight_layout()
                plt.savefig(self.logger.run_dir / f'confusion_matrix_epoch_{epoch}.png')
                plt.close()

            confmat.reset()
            roc_auc.reset()
            precision.reset()
            recall.reset()
            f1.reset()

        if self.rank == 0:
            print(eval_stat)
            print('Evaluation Completed.')

        return avg_loss, eval_stat