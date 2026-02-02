import torch
import torch.nn as nn
import timm
from copy import deepcopy
from library.config import Config


class DogCatClassifier(nn.Module):
    """
    Dog vs Cat Classifier using a ConvNeXt backbone.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(DogCatClassifier, self).__init__()

        # Create the model using timm
        # num_classes=1 results in a single output unit for binary classification
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Raw logits.
        """
        return self.model(x)


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy of the model that is updated as a moving average
    of the training model's weights.
    """

    def __init__(self, model, decay=Config.EMA_DECAY):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for the moving average.
        """
        self.decay = decay
        # Create a shadow copy of the model
        self.ema_model = deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for the shadow model to save memory/compute
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters based on the current model.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters
            for ema_param, param in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

            # Update buffers (e.g., running mean/var in BatchNorm, though ConvNeXt uses LayerNorm)
            # We simply copy buffers to keep them in sync
            for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)

    def get_model(self):
        """
        Returns the shadow model.
        """
        return self.ema_model
