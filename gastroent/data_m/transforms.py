from typing import Any
import albumentations as A
import kornia.augmentation as K
from albumentations.pytorch import ToTensorV2
import cv2

from utils_m.utils import to_hw

def get_transforms(prep: dict[str, Any], aug: dict[str, Any] | None = None, use_gpu: bool = False, classification: bool = True) -> K.AugmentationSequential | A.Compose:
    """
    Creates an image transformation/augmentation pipeline.

    CPU:
        Albumentations. Expects NumPy images in HWC format and returns
        NumPy images.

    GPU:
        Kornia. Expects torch tensors in CHW format and operates directly
        on the GPU.

    Args:
        prep (dict[str, Any]): Parameters used in image preprocessing.
            - resize_size (int | tuple): Target dimensions for the initial resize.
            - interpolation_type (str): Method used for resizing ('bilinear', 'bicubic', 'nearest').
            - center_crop (int | tuple): Dimensions for center crop.
            - mean (list | tuple): Channel means for tensor normalization.
            - std (list | tuple): Channel standard deviations for tensor normalization.
        aug (dict | None, optional): Parameters for dynamic, randomized data augmentation, with probabilities.
            Defaults to None.
            - hsv (dict): Ranges for photometric transforms ('hue', 'saturation', 'brightness').
            - rotation (list | tuple): Range of degrees for random rotations (e.g., [-30, 30]).
            - translation (list | tuple): Maximum spatial shift limits for the x and y axes.
                CPU expects pixel shift, GPU fractions of image dimensions.
            - scale (list | tuple): Lower and upper bounds for random scaling (e.g., [0.8, 1.2]).
            - interpolation_type (str): Interpolation method used during affine transforms ('bilinear', 'bicubic', 'nearest').
            - horizontal_flip (bool): Enabling the horizontal flip.
            - vertical_flip (bool): Enabling the vertical flip.
            - fill (int | str): Pixel value used to fill empty space created by affine shifts.
                For GPU transforms: 'zeros', 'border', 'reflection'. For CPU integer value.
            - random_crop (int | tuple): Target output size for random crops.
            - probs (dict): Execution probabilities (0.0 to 1.0) mapping to specific transforms ('hsv', 'affine', horizontal_flip', 'vertical_flip', 'random_crop).
        use_gpu (bool, optional): Selection between running transformations on 
            GPU or CPU. CPU version runs on Albumentations, GPU runs on Kornia.
            Defaults to False.
        classification (bool, optional): In segmentation, mask requires different preprocessing.
            Defaults to True.

    Returns:
        (K.AugmentationSequential | A.Compose): Preprocessing pipeline.

    Raises:
        ValueError: If 'center_crop' in `prep` is not provided.

    Images entering this pipeline should be float32 tensors in [0, 1].
    """

    interpolation_map = {'bilinear': cv2.INTER_LINEAR,
                         'bicubic': cv2.INTER_CUBIC,
                         'nearest': cv2.INTER_NEAREST}
    ops = []

    center_crop = prep.get('center_crop', None)

    if center_crop is None:
        raise ValueError('Expected center_crop in the prep dictionary.')

    if aug:
        probs = aug.get('probs', {})
        hsv =  aug.get('hsv', {})
        h_flip =  aug.get('horizontal_flip', False)
        v_flip = aug.get('vertical_flip', False)
        random_crop =  aug.get('random_crop')
        affine_interpolation = aug.get('interpolation_type', 'bilinear')

        if affine_interpolation not in interpolation_map:
            raise ValueError(f'Unknown interpolation: {affine_interpolation}. Expected one of {list(interpolation_map)}.')

    # CPU - Albumentations
    if not use_gpu:
        if aug:
            if hsv:
                ops.append(A.ColorJitter(hue = hsv.get('hue', 0),
                                         saturation = hsv.get('saturation', 0),
                                         brightness = hsv.get('brightness', 0),
                                         p = probs.get('hsv', 0.5)))

            fill_val = aug.get('fill', 0)

            if fill_val == 'reflection':
                mode = cv2.BORDER_REFLECT
                fill = 0
            elif fill_val == 'border':
                mode = cv2.BORDER_REPLICATE
                fill = 0
            else:
                mode = cv2.BORDER_CONSTANT
                fill = fill_val

            ops.append(A.Affine(rotate = aug.get('rotation', 0),
                                translate_px = aug.get('translation'),
                                scale = aug.get('scale'),
                                fill = fill,
                                mode = mode,
                                interpolation = interpolation_map[affine_interpolation],
                                p = probs.get('affine', 0.5)))
            
            if h_flip:
                ops.append(A.HorizontalFlip(p = probs.get('horizontal_flip', 0.5)))
    
            if v_flip:
                ops.append(A.VerticalFlip(p = probs.get('vertical_flip', 0.5)))
    
            if random_crop:
                random_crop = to_hw(random_crop)
                ops.append(A.RandomCrop(height = random_crop[0], width = random_crop[1], p = probs.get('random_crop', 0.5))) 

        center_crop = to_hw(center_crop)
        ops.extend([A.CenterCrop(height = center_crop[0], width = center_crop[1], p = 1.0),
                    A.Normalize(mean = prep['mean'], std = prep['std'], max_pixel_value = 1.0),
                    ToTensorV2()])
        return A.Compose(ops, additional_targets = {'mask': 'mask'} if not classification else None)

    # GPU - Kornia
    if aug:
        if hsv:
            ops.append(K.ColorJitter(hue = hsv.get('hue', 0),
                                     saturation = hsv.get('saturation', 0),
                                     brightness = hsv.get('brightness', 0),
                                     p = probs.get('hsv', 0.5)))
            
        ops.append(K.RandomAffine(degrees =  aug.get('rotation', 0),
                                  translate =  aug.get('translation'),
                                  scale = aug.get('scale'),
                                  padding_mode =  aug.get('fill', 'zeros'), # zeros, border, reflection
                                  resample =  affine_interpolation,
                                  p = probs.get('affine', 0.5)))
      
        if h_flip:
            ops.append(K.RandomHorizontalFlip(p = probs.get('horizontal_flip', 0.5)))

        if v_flip:
            ops.append(K.RandomVerticalFlip(p = probs.get('vertical_flip', 0.5)))

        if random_crop:
            ops.append(K.RandomCrop(size = random_crop, p = probs.get('random_crop', 0.5))) 

    ops.extend([K.CenterCrop(center_crop),
                K.Normalize(mean = prep['mean'], std = prep['std'])])
    return K.AugmentationSequential(*ops, data_keys = ['input'] if classification else ['input', 'mask'])