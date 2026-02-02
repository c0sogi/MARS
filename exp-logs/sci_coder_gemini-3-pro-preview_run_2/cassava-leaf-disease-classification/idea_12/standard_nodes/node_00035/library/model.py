import copy
import torch
import timm
from library.config import Config


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model weights.
    Maintains a shadow model that updates slowly based on the training model.
    Includes functionality to reset weights for curriculum learning phases.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=Config.DEVICE):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay rate for EMA (default from Config).
            device (str): Device to store the EMA model on.
        """
        self.decay = decay
        self.device = device

        # Create a deep copy of the model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.to(self.device)

        # Disable gradients for the EMA model to save memory/compute
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters using the current model.
        Formula: shadow = decay * shadow + (1 - decay) * current
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()

            for k, v in msd.items():
                if k in esd:
                    v = v.to(self.device)
                    # Apply EMA to floating point parameters/buffers
                    if esd[k].dtype.is_floating_point:
                        esd[k].mul_(self.decay).add_(v, alpha=(1 - self.decay))
                    # Directly copy integer parameters (e.g., num_batches_tracked)
                    else:
                        esd[k].copy_(v)

    def set_weights(self, model):
        """
        Hard reset of the EMA model weights to match the current model.
        Used at the beginning of new training phases (e.g., resolution change)
        to eliminate lag from previous phases.
        """
        self.ema_model.load_state_dict(model.state_dict())


def get_model(pretrained=True):
    """
    Instantiates the ConvNeXt Small model using timm.

    Args:
        pretrained (bool): Whether to load ImageNet-1k pretrained weights.

    Returns:
        torch.nn.Module: The configured model on the correct device.
    """
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=Config.NUM_CLASSES,
        drop_path_rate=Config.DROP_PATH_RATE,
    )

    model.to(Config.DEVICE)
    return model
