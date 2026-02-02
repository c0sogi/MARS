import torch
import torch.nn as nn
import copy
import timm
from library.config import Config


def get_model(device=None):
    """
    Creates and returns the ConvNeXt-Small model configured for binary classification.

    Args:
        device (torch.device, optional): The device to load the model onto.
                                         Defaults to Config.DEVICE.

    Returns:
        torch.nn.Module: The configured PyTorch model.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    print(f"Initializing model: {Config.MODEL_NAME}")

    # Create the model using timm
    # num_classes=1 sets the final linear layer to output a single logit
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        drop_rate=Config.DROP_RATE,
        drop_path_rate=Config.DROP_PATH_RATE,
    )

    # Move model to the specified device
    model = model.to(device)

    return model


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model weights.
    Maintains a 'shadow' model whose parameters are a weighted average of the
    training model's parameters over time. This often leads to better generalization.
    """

    def __init__(self, model, decay: float = Config.EMA_DECAY, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for the moving average (default: 0.9999).
            device (torch.device, optional): Device for the shadow model.
        """
        self.decay = decay

        # Create a deep copy of the model to serve as the shadow model
        self.ema_model = copy.deepcopy(model)

        # Ensure the shadow model is in evaluation mode
        self.ema_model.eval()

        # Move to device if specified, otherwise keep on same device as source model
        if device:
            self.ema_model.to(device)

        # Disable gradients for the shadow model to save memory and computation
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Updates the shadow model parameters using the current model's parameters.
        Formula: shadow_weight = decay * shadow_weight + (1 - decay) * current_weight

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            # Iterate over state_dict to handle both parameters and buffers (e.g., BatchNorm stats)
            msd = model.state_dict()
            esd = self.ema_model.state_dict()

            for name, param in msd.items():
                if name in esd:
                    ema_param = esd[name]

                    # Only apply EMA to floating point parameters/buffers
                    if param.dtype.is_floating_point:
                        # In-place update for efficiency
                        # ema_param = ema_param * decay + param * (1 - decay)
                        ema_param.mul_(self.decay).add_(param, alpha=1.0 - self.decay)
                    else:
                        # For integer types (e.g., num_batches_tracked), just copy the value
                        ema_param.copy_(param)

    def get_model(self):
        """
        Returns the shadow (EMA) model.

        Returns:
            torch.nn.Module: The EMA model.
        """
        return self.ema_model
