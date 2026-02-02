import copy
import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_logger

logger = get_logger("model_module")


class AnimalModel(nn.Module):
    """
    Neural network model for Animal Classification.
    Wraps a timm backbone (ConvNeXt-Small) and adapts the head for the specific number of classes.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        num_classes: int = Config.NUM_CLASSES,
        pretrained: bool = True,
    ):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(AnimalModel, self).__init__()
        logger.info(
            f"Initializing model: {model_name} | Pretrained: {pretrained} | Classes: {num_classes}"
        )

        # Create the model using timm
        # timm handles the replacement of the classification head (fc/classifier) automatically
        # when num_classes is provided.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Logits.
        """
        return self.backbone(x)


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Maintains a shadow copy of the model that updates slowly, often leading to better generalization.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = Config.EMA_DECAY,
        device: torch.device = None,
    ):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor (beta) for the moving average.
                           Higher values (e.g., 0.9999) mean smoother, slower updates.
            device (torch.device): Device to store the EMA model on.
        """
        self.decay = decay

        # Create a deep copy of the model
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        # Move to appropriate device
        if device is None:
            device = Config.DEVICE
        self.ema.to(device)

        # Disable gradients for the EMA model to save memory and computation
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module):
        """
        Update the EMA parameters.
        Formula: theta_ema = decay * theta_ema + (1 - decay) * theta_current

        Args:
            model (nn.Module): The current training model from which to update.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, param in msd.items():
                if name in esd:
                    ema_v = esd[name]
                    model_v = param.detach()

                    # Update floating point parameters (weights, biases)
                    if ema_v.dtype.is_floating_point:
                        ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
                    else:
                        # For non-floating point parameters (e.g., num_batches_tracked in BatchNorm),
                        # we copy the value directly without averaging.
                        ema_v.copy_(model_v)
