import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Any

from models_m.classification_models import ClassificationModel
from models_m.segmentation_models import SegmentationModel
from data_m.dataset_classification import CLASS_TO_IDX

RGB_PREPROCESSING = {'resize_size': 256,
                     'interpolation_type': v2.InterpolationMode.BILINEAR,
                     'center_crop': 224,
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

def test_classification(model: ClassificationModel, experiment: str, run_index: int, epoch: int, images: str | list[str], prep: dict[str, Any] = None, rgb: bool = True, labels: str | list[str] = None) -> None:
    """
    Testing the model on individual or group of images.

    Args:
        model (ClassificationModel): Any of the existing models for classification.
        experiment (str): Name of the experiment, used in accesssing the saved model.
        run_index (int): Number of the run, used in accessing the saved model.
        epoch (int): Number of the epoch, used in accessing the saved model.
        images (str | list[str]): Paths to images.
        prep (dict[str, Any] | None, optional): Dicitonary with parameters for preprocessing the images.
            Defaults to None. When not using RGB_PREPROCESSING, another must be provided.
        rgb (bool, optional): RGB or grayscale mode. Defaults to True.
        labels (str | list[str] | None, optional): List of labels for the images. Used to
            compare model results with groundtruth. Defaults to None.

    Raises:
        ValueError: If RGB_PREPROCESSING is not used, another dictionary must be provided.
    """

    model = model.model

    base_dir = Path(__file__).resolve().parents[2]
    file_path = base_dir / 'models' / experiment / f'run_{run_index:03d}' / 'checkpoints' / f'epoch_{epoch:03d}.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(file_path, map_location = device)

    current_model = model.__class__.__name__
    loaded_model = checkpoint.get('model_name', 'Unknown')

    if current_model != loaded_model:
        raise AttributeError(f'Models do not match. Loaded: {loaded_model}, used: {current_model}.')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    if prep is None and rgb is True:
        prep = RGB_PREPROCESSING
    else:
        raise ValueError('Preprocessing dictionary is required.')

    transform_pipeline = v2.Compose([v2.ToImage(),
                                     v2.Resize(size = prep['resize_size'], interpolation = prep.get('interpolation_type', v2.InterpolationMode.BILINEAR)),
                                     v2.CenterCrop(size = prep['center_crop']),
                                     v2.ToDtype(torch.float32, scale = True),
                                     v2.Normalize(mean = prep['mean'], std = prep['std'])])
    
    open_mode = 'RGB' if rgb else 'L'
    image_dir = base_dir / 'data'

    if isinstance(images, str):
        images = [images]

    if isinstance(labels, str):
        labels = [labels]
    elif labels is None:
        labels = [None] * len(images)

    IDX_TO_CLASS = {}

    for class_name, idx in CLASS_TO_IDX.items():
        if idx in IDX_TO_CLASS:
            IDX_TO_CLASS[idx] += f' / {class_name}'
        else:
            IDX_TO_CLASS[idx] = class_name

    with torch.no_grad():
        for img_name, label in zip(images, labels):
            img_path = image_dir / img_name
            image = Image.open(img_path).convert(open_mode)
            input_tensor = transform_pipeline(image).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim = 1)
            top3_probs, top3_indices = torch.topk(probabilities, k = 3, dim = 1)
            
            print(f'Predictions for {img_name}:')

            for i in range(3):
                prob = top3_probs[0][i].item() * 100
                idx = top3_indices[0][i].item()
                class_name = IDX_TO_CLASS[idx]

                if not i and label is not None:
                    valid_names = class_name.split(' / ')
                    if label in valid_names:
                        print(f'Correct prediction: {class_name}.')
                    else:
                        print(f'Prediction incorrect. Label: {label}, prediction: {class_name}.')
                
                print(f'  {i + 1}. {class_name}: {prob:.2f}%')
                
            print('-' * 30)

def test_segmentation(model: SegmentationModel, experiment: str, run_index: int, epoch: int, data_folder: str, images: str | list[str], prep: dict[str, Any], rgb: bool = True) -> None:
    """
    Testing the model on individual or group of images.

    Args:
        model (SegmentationModel): Any of the existing models for segmentation.
        experiment (str): Name of the experiment, used in accesssing the saved model.
        run_index (int): Number of the run, used in accessing the saved model.
        epoch (int): Number of the epoch, used in accessing the saved model.
        data_folder (str): Folder with subfolders 'images' and optionally 'masks'.
        images (str | list[str]): Paths to images. Masks should have exactly the same.
        prep (dict[str, Any] | None, optional): Dicitonary with parameters for preprocessing the images.
            Defaults to None. When not using RGB_PREPROCESSING, another must be provided.
        rgb (bool, optional): RGB or grayscale mode. Defaults to True.
    """

    model = model.model

    base_dir = Path(__file__).resolve().parents[2]
    file_path = base_dir / 'models' / experiment / f'run_{run_index:03d}' / 'checkpoints' / f'epoch_{epoch:03d}.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(file_path, map_location = device)

    current_model = model.__class__.__name__
    loaded_model = checkpoint.get('model_name', 'Unknown')
    
    if current_model != loaded_model:
        raise AttributeError(f'Models do not match. Loaded: {loaded_model}, used: {current_model}.')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    if prep is None and rgb is True:
            prep = RGB_PREPROCESSING
    else:
        raise ValueError('Preprocessing dictionary is required.')

    transform_pipeline = v2.Compose([v2.ToImage(),
                                     v2.Resize(size = prep['resize_size'], interpolation = prep.get('interpolation_type', v2.InterpolationMode.BILINEAR)),
                                     v2.CenterCrop(size = prep['center_crop']),
                                     v2.ToDtype(torch.float32, scale = True),
                                     v2.Normalize(mean = prep['mean'], std = prep['std'])])
    mask_transform = v2.Compose([v2.ToImage(),
                                 v2.Resize(size = prep['resize_size'], interpolation = v2.InterpolationMode.NEAREST),
                                 v2.CenterCrop(size = prep['center_crop']),
                                 v2.ToDtype(torch.float32, scale = True)])
        
    open_mode = 'RGB' if rgb else 'L'
    
    data_path = base_dir / 'data' / data_folder
    image_dir = data_path / 'images'
    mask_dir = data_path / 'masks'

    if isinstance(images, str):
        images = [images]

    with torch.no_grad():
        for img_name in images:
            mask_name = img_name
            
            img_path = image_dir / img_name
            image = Image.open(img_path).convert(open_mode)
            input_tensor = transform_pipeline(image).unsqueeze(0).to(device)
            
            output = model(input_tensor)

            preds = torch.argmax(output, dim = 1, keepdim = True).float()
            pred_np = preds.squeeze().cpu().numpy()

            has_mask = False
            mask_path = mask_dir / mask_name

            if mask_path.exists():
                has_mask = True
            else:
                print(f'Warning: Mask {mask_name} not found at {mask_path}. Skipping metrics.')
            
            dice_score, iou_score = 0.0, 0.0
            
            if has_mask:
                mask = Image.open(mask_path).convert('L')
                mask_tensor = mask_transform(mask).unsqueeze(0).to(device)
                mask_tensor = (mask_tensor > 0.5).float()
                mask_np = mask_tensor.squeeze().cpu().numpy()

                pred_flat = preds.view(-1)
                mask_flat = mask_tensor.view(-1)

                intersection = (pred_flat * mask_flat).sum()
                union = pred_flat.sum() + mask_flat.sum() - intersection

                dice_score = (2.0 * intersection) / (pred_flat.sum() + mask_flat.sum() + 1e-8)
                iou_score = intersection / (union + 1e-8)

                print(f'Metrics for {img_name} - Dice: {dice_score.item():.4f} | IoU: {iou_score.item():.4f}.')
            else:
                mask_np = None
                print(f'Processed {img_name}. No ground truth found.')

            if has_mask: 
                fig, axes = plt.subplots(1, 3, figsize = (10, 5)) 
                axes[0].imshow(image) 
                axes[0].set_title('Image') 
                axes[1].imshow(mask_np, cmap = 'gray') 
                axes[1].set_title('Ground Truth') 
                axes[2].imshow(pred_np, cmap = 'gray') 
                axes[2].set_title('Output') 
            else: 
                fig, axes = plt.subplots(1, 3, figsize = (10, 5)) 
                axes[0].imshow(image) 
                axes[0].set_title('Image') 
                axes[1].text(0.5, 0.5, 'Ground truth not available', ha = 'center', va = 'center', fontsize = 14) 
                axes[1].set_title('Ground Truth') 
                axes[2].imshow(pred_np, cmap = 'gray') 
                axes[2].set_title('Output') 

            for ax in axes: 
                ax.axis('off') 
                
            fig.suptitle(img_name) 
            plt.tight_layout() 
            plt.show() 
            plt.close(fig)