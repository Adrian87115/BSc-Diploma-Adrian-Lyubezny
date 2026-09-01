import torch
import torch.nn as nn
from torchvision.models import (resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, 
                                resnet101, ResNet101_Weights, resnet152, ResNet152_Weights,
                                convnext_tiny, ConvNeXt_Tiny_Weights, convnext_small, ConvNeXt_Small_Weights, 
                                convnext_base, ConvNeXt_Base_Weights, convnext_large, ConvNeXt_Large_Weights,
                                swin_t, Swin_T_Weights, swin_s, Swin_S_Weights, swin_b, Swin_B_Weights,
                                swin_v2_t, Swin_V2_T_Weights, swin_v2_s, Swin_V2_S_Weights, swin_v2_b, Swin_V2_B_Weights)

class ClassificationModel(nn.Module):
    """
    A wrapper class for image classification neural networks.

    Validates input parameters and dynamically loads the specified model 
    architecture for image classification tasks.

    Using pretrained models requires proper data preprocessing.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        num_classes (int): Number of target classes for classification. Must be >= 2.
        m_type (str): The architecture type of the model, must be selected from the predefined list.
        pretrained (bool, optional): Whether to load pretrained ImageNet weights. 
            Defaults to False.

    Raises:
        ValueError: If `in_channels` is not 1 or 3, or if `num_classes` is less than 2.
    """

    def __init__(self, in_channels: int, num_classes: int, m_type: str, pretrained: bool = False):
        super(ClassificationModel, self).__init__()

        if in_channels not in [1, 3]:
            raise ValueError('Incorrect input channel size, allowed: 1, 3.')

        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('Number of classes must be an integer >= 2.')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.m_type = m_type
        self.pretrained = pretrained

        self.model = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images, shape 
                (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Logits (raw predictions) for each class, shape 
                (batch_size, num_classes).
        """

        if self.model is None:
            raise NotImplementedError('self.model was not initialized.')
        
        return self.model(x)

class ResNetModel(ClassificationModel):
    """
    Creates a ResNet instance for image classification.

    Models can be trained from scratch or load pretrained ImageNet weights.
    Available ResNet architectures ('m_type'): '18', '34', '50', '101', '152'.

    Available pretrained models:
    18, 34: Only IMAGENET1K_V1
    50, 101, 152: IMAGENET1K_V1, IMAGENET1K_V2

    Preprocessing:   
    IMAGENET1K_V1 and IMAGENET1K_V2:
        - accepts PIL.Image batched (B, C, H, W) and single (C, H, W) torch.Tensor
        IMAGENET1K_V1:
            - images resized to resize_size = [256]
        IMAGENET1K_V2:
            - images resized to resize_size = [232]
        - interpolation = InterpolationMode.BILINEAR
        - central crop of crop_size = [224]
        - values rescaled to [0.0, 1.0]
        - normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

    More details: https://docs.pytorch.org/vision/main/models/resnet.html

    Args:
        in_channels (int): Number of input channels (1 or 3). Passed to base class.
        num_classes (int): Target number of classes for the final layer. Passed to base.
        m_type (str): The specific ResNet depth ('18', '34', '50', '101', '152'). Passed to base.
        pretrained (bool, optional): Whether to use ImageNet weights. Defaults to False. 
            Passed to base.
        weights_ver (int, optional): Version of ImageNet weights to use (1 or 2). 
            If 2 is requested but unavailable, falls back to 1. Defaults to 1.

    Raises:
        ValueError: If `m_type` is not from the predefined pool, or if `weights_ver` is not 1 or 2.
    """

    def __init__(self, *args, weights_ver: int = 1, **kwargs):
        super(ResNetModel, self).__init__(*args, **kwargs)
        
        resnet_type_table = {'18': (resnet18, ResNet18_Weights),
                             '34': (resnet34, ResNet34_Weights),
                             '50': (resnet50, ResNet50_Weights),
                             '101': (resnet101, ResNet101_Weights),
                             '152': (resnet152, ResNet152_Weights)}

        if self.m_type not in resnet_type_table.keys():
            raise ValueError(f'Chosen model does not exist, allowed: {resnet_type_table.keys()}')

        if weights_ver not in [1, 2]:
            raise ValueError(f'Incorrect pretrained model, allowed: 1, 2.')

        self.weights_ver = weights_ver
        model_func, weights = resnet_type_table[self.m_type]

        if self.pretrained:
            if self.m_type in ['18', '34'] and weights_ver == 2:
                print(f'weights_ver == 2 is not available for ResNet{self.m_type}. Switching to weights_ver = 1')
                weights_ver = 1
            
            weights_choice = weights.IMAGENET1K_V1 if weights_ver == 1 else weights.IMAGENET1K_V2
            self.model = model_func(weights = weights_choice)
        else:
            self.model = model_func(weights = None)

        # Channel Adjustment
        if self.in_channels != 3:
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(in_channels = self.in_channels,
                                 out_channels = old_conv.out_channels,
                                 kernel_size = old_conv.kernel_size,
                                 stride = old_conv.stride,
                                 padding = old_conv.padding,
                                 bias = (old_conv.bias is not None))

            if self.pretrained and self.in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim = 1, keepdim = True))    # Preserving the learned features

            self.model.conv1 = new_conv

        # New Classification Head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

class ConvNeXtModel(ClassificationModel):
    """
    Creates a ConvNeXt instance for image classification.

    Models can be trained from scratch or load pretrained ImageNet weights.
    Available ConvNeXt architectures ('m_type'): 'T', 'S', 'B', 'L'.

    Available pretrained models:
    T, S, B, L: IMAGENET1K_V1

    Preprocessing:   
    IMAGENET1K_V1:
        - accepts PIL.Image batched (B, C, H, W) and single (C, H, W) torch.Tensor
        T:
            - images resized to resize_size = [236]
        S:
            - images resized to resize_size = [230]
        B, L:
            - images resized to resize_size = [232]
        - interpolation = InterpolationMode.BILINEAR
        - central crop of crop_size = [224]
        - values rescaled to [0.0, 1.0]
        - normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

    More details: https://docs.pytorch.org/vision/main/models/convnext.html

    Args:
        in_channels (int): Number of input channels (1 or 3). Passed to base class.
        num_classes (int): Target number of classes for the final layer. Passed to base.
        m_type (str): The specific ConvNeXt depth ('T', 'S', 'B', 'L'). Passed to base.
        pretrained (bool, optional): Whether to use ImageNet weights. Defaults to False. 
            Passed to base.

    Raises:
        ValueError: If `m_type` is not from the predefined pool.
    """
    
    def __init__(self, *args, **kwargs):
        super(ConvNeXtModel, self).__init__(*args, **kwargs)

        convnext_type_table = {'T': (convnext_tiny, ConvNeXt_Tiny_Weights),
                               'S': (convnext_small, ConvNeXt_Small_Weights),
                               'B': (convnext_base, ConvNeXt_Base_Weights),
                               'L': (convnext_large, ConvNeXt_Large_Weights)}

        if self.m_type not in convnext_type_table.keys():
            raise ValueError(f'Chosen model does not exist, allowed: {convnext_type_table.keys()}.')

        model_func, weights = convnext_type_table[self.m_type]
        self.model = model_func(weights = weights.IMAGENET1K_V1 if self.pretrained else None)

        # Channel Adjustment
        if self.in_channels != 3:
            old_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(in_channels = self.in_channels,
                                 out_channels = old_conv.out_channels,
                                 kernel_size = old_conv.kernel_size,
                                 stride = old_conv.stride,
                                 padding = old_conv.padding,
                                 bias = (old_conv.bias is not None))

            if self.pretrained and self.in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim = 1, keepdim = True))    # Preserving the learned features

            self.model.features[0][0] = new_conv
            
        # New Classification Head
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, self.num_classes)

class SwinTransformerModel(ClassificationModel):
    """
    Creates a SwinTransformer or SwinTransformer V2 instance for image classification.

    Models can be trained from scratch or load pretrained ImageNet weights.
    Available SwinTransformer architectures ('m_type'): 'T', 'S', 'B', 'T2', 'S2', 'B2'.

    Available pretrained models:
    T, S, B, T2, S2, B2: IMAGENET1K_V1

    Preprocessing:   
    IMAGENET1K_V1:
        T:
            - images resized to resize_size = [232]
        S:
            - images resized to resize_size = [246]
        B:
            - images resized to resize_size = [238]
        T2, S2:
            - images resized to resize_size = [260]
        B2:
            - images resized to resize_size = [272]
        - interpolation = InterpolationMode.BICUBIC
        T, S, B:
            - central crop of crop_size = [224]
        T2, S2, B2:
            - central crop of crop_size = [256]
        - values rescaled to [0.0, 1.0]
        - normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

    More details: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_s.html#torchvision.models.swin_s

    Args:
        in_channels (int): Number of input channels (1 or 3). Passed to base class.
        num_classes (int): Target number of classes for the final layer. Passed to base.
        m_type (str): The specific SwinTransformer depth ('T', 'S', 'B', 'T2', 'S2', 'B2'). Passed to base.
        pretrained (bool, optional): Whether to use ImageNet weights. Defaults to False. 
            Passed to base.

    Raises:
        ValueError: If `m_type` is not from the predefined pool.
    """
    
    def __init__(self, *args, **kwargs):
        super(SwinTransformerModel, self).__init__(*args, **kwargs)
                  
        swin_trans_type_table = {'T': (swin_t, Swin_T_Weights),
                                 'S': (swin_s, Swin_S_Weights),
                                 'B': (swin_b, Swin_B_Weights),
                                 'T2': (swin_v2_t, Swin_V2_T_Weights),
                                 'S2': (swin_v2_s, Swin_V2_S_Weights),
                                 'B2': (swin_v2_b, Swin_V2_B_Weights),}

        if self.m_type not in swin_trans_type_table.keys():
            raise ValueError(f'Chosen model does not exist, allowed: {swin_trans_type_table.keys()}.')
        
        model_func, weights = swin_trans_type_table[self.m_type]
        self.model = model_func(weights = weights.DEFAULT if self.pretrained else None)

        # Channel Adjustment
        if self.in_channels != 3:
            old_conv = self.model.features[0][0]
            
            new_conv = nn.Conv2d(in_channels = self.in_channels,
                                 out_channels = old_conv.out_channels,
                                 kernel_size = old_conv.kernel_size,
                                 stride = old_conv.stride,
                                 padding = old_conv.padding,
                                 bias = (old_conv.bias is not None))

            if self.pretrained and self.in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim = 1, keepdim = True))    # Preserving the learned features

            self.model.features[0][0] = new_conv

        # New Classification Head
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, self.num_classes)