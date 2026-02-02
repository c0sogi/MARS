import copy
import torch
import torch.nn as nn
import timm
from library.config import Config


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a shadow copy of the model that is updated via a moving average
    of the training model's weights. This creates a temporal ensemble that
    often leads to better generalization and stability.
    """

    def __init__(self, model, decay=None):
        """
        Args:
            model (nn.Module): The source model to track.
            decay (float, optional): The decay rate for the moving average.
                                     If None, defaults to Config.EMA_DECAY.
        """
        self.decay = decay if decay is not None else Config.EMA_DECAY

        # Create a deep copy of the model to serve as the shadow model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for the shadow model parameters to save memory/compute
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters using the current training model parameters.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update learnable parameters
            # We assume the model architectures are identical, so we can zip safely
            for (name, param), (ema_name, ema_param) in zip(
                model.named_parameters(), self.ema_model.named_parameters()
            ):
                if param.requires_grad:
                    # Formula: ema_param = decay * ema_param + (1 - decay) * current_param
                    ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

            # Update buffers (e.g., running statistics for normalization layers)
            # These are copied directly to keep the EMA model up to date with the training state
            for (name, buffer), (ema_name, ema_buffer) in zip(
                model.named_buffers(), self.ema_model.named_buffers()
            ):
                ema_buffer.copy_(buffer)


def get_model(pretrained=True):
    """
    Creates and returns the ConvNeXt-Tiny model.

    Configuration:
    - Backbone: convnext_tiny.fb_in1k
    - Pooling: Global Average Pooling (GAP) via global_pool='avg'
    - Head: LayerNorm -> Flatten -> Linear (1 class) (Standard timm ConvNeXt head)
    - Regularization: Stochastic Depth (Drop Path)

    Args:
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    # timm.create_model handles the backbone, pooling, and head construction.
    # global_pool='avg' ensures Global Average Pooling is used.
    # The default head for ConvNeXt in timm includes the pre-classifier LayerNorm.
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=Config.NUM_CLASSES,
        drop_path_rate=Config.DROP_PATH_RATE,
        global_pool="avg",
    )

    return model
