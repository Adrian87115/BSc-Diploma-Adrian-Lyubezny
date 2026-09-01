import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):
    """
    A wrapper class for image segmentation neural networks.

    Validates input parameters and dynamically loads the specified model 
    architecture for image segmentation tasks.

    Using pretrained models requires proper data preprocessing.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        encoder (str): Selected encoder and decoder used in the architecture, 
            must be selected from the predefinded list.
        pretrained (bool, optional): Whether to load pretrained weights. 
            Defaults to False.
        weights_ver (str, optional): Selection of predefined weights. Defaults to 'imagenet'.

    Raises:
        ValueError: If `in_channels` is not 1 or 3, if the encoder is not recognized, 
            or if the requested pretrained weights are not available for the encoder.

    More about encoders: https://smp.readthedocs.io/en/latest/encoders.html
    """
    
    def __init__(self, in_channels: int, encoder: str, pretrained: bool = False, weights_ver: str = 'imagenet'):
        super().__init__()

        if in_channels not in [1, 3]:
            raise ValueError('Incorrect input channel size, allowed: 1, 3.')

        self.in_channels = in_channels
        self.encoder = encoder
        self.pretrained = pretrained
        self.weights_ver = weights_ver

        # Encoders with their pretrained weights. 
        self.encoders = {'resnet18': ['imagenet', 'ssl', 'swsl'],                        # 11M
                         'resnet34': ['imagenet'],                                       # 21M
                         'resnet50': ['imagenet', 'ssl', 'swsl'],                        # 23M
                         'resnet101': ['imagenet'],                                      # 42M
                         'resnet152': ['imagenet'],                                      # 58M
                         
                         'resnext50_32x4d': ['imagenet', 'ssl', 'swsl'],                 # 22M
                         'resnext101_32x4d': ['ssl', 'swsl'],                            # 42M
                         'resnext101_32x8d': ['imagenet', 'instagram', 'ssl', 'swsl'],   # 86M
                         'resnext101_32x16d': ['instagram', 'ssl', 'swsl'],              # 191M
                         'resnext101_32x32d': ['instagram'],                             # 466M
                         'resnext101_32x48d': ['instagram'],                             # 826M
                         
                         'mobilenet_v2': ['imagenet'],                                     # 2M
                         
                         'efficientnet-b0': ['imagenet', 'advprop'],                     # 4M
                         'efficientnet-b1': ['imagenet', 'advprop'],                     # 6M
                         'efficientnet-b2': ['imagenet', 'advprop'],                     # 7M
                         'efficientnet-b3': ['imagenet', 'advprop'],                     # 10M
                         'efficientnet-b4': ['imagenet', 'advprop'],                     # 17M
                         'efficientnet-b5': ['imagenet', 'advprop'],                     # 28M
                         'efficientnet-b6': ['imagenet', 'advprop'],                     # 40M
                         'efficientnet-b7': ['imagenet', 'advprop'],                     # 63M
                         
                         'mit_b0': ['imagenet'],                                         # 3M
                         'mit_b1': ['imagenet'],                                         # 13M
                         'mit_b2': ['imagenet'],                                         # 24M
                         'mit_b3': ['imagenet'],                                         # 44M
                         'mit_b4': ['imagenet'],                                         # 60M
                         'mit_b5': ['imagenet']}                                         # 81M             

        if encoder not in self.encoders:
            raise ValueError(f'Chosen encoder does not exist, allowed: {self.encoders.keys()}.')
        
        if pretrained and weights_ver not in self.encoders[encoder]:
            raise ValueError(f'Invalid pretrained weights for {encoder}, available: {self.encoders[encoder]}.')

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

class UNetPlusPlusModel(SegmentationModel):
    """
    Creates a UNetPlusPlus instance for image segmentation.
    Model details: https://smp.readthedocs.io/en/latest/models.html#unetplusplus

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        encoder (str): Selected encoder and decoder used in the architecture, 
            must be selected from the predefinded list.
        pretrained (bool, optional): Whether to load pretrained weights. 
            Defaults to False.
        weights_ver (str, optional): Selection of predefined weights. Defaults to 'imagenet'.
    
    Raises:
        ValueError: If `in_channels` is not 1 or 3, if the encoder is not recognized, 
            or if the requested pretrained weights are not available for the encoder.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = smp.UnetPlusPlus(encoder_name = self.encoder, encoder_weights = self.weights_ver if self.pretrained else None, in_channels = self.in_channels, classes = 2)

class DeepLabV3PlusModel(SegmentationModel):
    """
    Creates a DeepLabV3Plus instance for image segmentation.
    Model details: https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        encoder (str): Selected encoder and decoder used in the architecture, 
            must be selected from the predefinded list.
        pretrained (bool, optional): Whether to load pretrained weights. 
            Defaults to False.
        weights_ver (str, optional): Selection of predefined weights. Defaults to 'imagenet'.
    
    Raises:
        ValueError: If `in_channels` is not 1 or 3, if the encoder is not recognized, 
            or if the requested pretrained weights are not available for the encoder.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = smp.DeepLabV3Plus(encoder_name = self.encoder, encoder_weights = self.weights_ver if self.pretrained else None, in_channels = self.in_channels, classes = 2)

class SegFormerModel(SegmentationModel):
    """
    Creates a SegFormer instance for image segmentation.
    Model details: https://smp.readthedocs.io/en/latest/models.html#segformer
    
    Although all encoders can work, suggested are Mix Vision Transformer (mit).

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            Passed to base class.
        encoder (str): Selected encoder used in the architecture. Passed to base class.
        pretrained (bool, optional): Whether to load pretrained weights. 
            Defaults to False. Passed to base class.
        weights_ver (str, optional): Selection of predefined weights. 
            Defaults to 'imagenet'. Passed to base class.
    
    Raises:
        ValueError: If `in_channels` is not 1 or 3, if the encoder is not recognized, 
            or if the requested pretrained weights are not available for the encoder.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.encoder.startswith('mit_b'):
            print(f'Suggested mit_b encoders! Used: {self.encoder}')
        
        self.model = smp.Segformer(encoder_name = self.encoder, encoder_weights = self.weights_ver if self.pretrained else None, in_channels = self.in_channels, classes = 2)