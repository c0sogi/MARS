import torch
import torch.nn as nn
from copy import deepcopy
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("ema.log")


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) of model parameters.
    Maintains a shadow copy of the model weights that updates as a moving average.
    This stabilizes the model and often leads to better generalization.
    """

    def __init__(self, model, decay=Config.ema_decay, device=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.999).
            device (torch.device): Device to store shadow parameters on.
                                   If None, uses the model's current device.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Determine device
        if device is None:
            # Try to infer device from the first parameter
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Register model parameters
        self.register()
        logger.info(
            f"ModelEMA initialized with decay={self.decay} on device={self.device}"
        )

    def register(self):
        """
        Initialize shadow parameters as a deep copy of the current model parameters.
        Only tracks parameters that require gradients.
        """
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Clone and detach to create a separate copy of the weights
                self.shadow[name] = param.data.clone().detach().to(self.device)

    @torch.no_grad()
    def update(self):
        """
        Update shadow parameters using the EMA formula:
        shadow = decay * shadow + (1 - decay) * current_param

        This should be called after every optimizer step.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in EMA shadow."

                # Update shadow weights in-place for efficiency
                # new_average = (1.0 - decay) * param.data + decay * shadow

                # We use the formula: shadow.mul_(decay).add_(param, alpha=1-decay)
                # This is mathematically equivalent to: shadow = shadow * decay + param * (1 - decay)

                shadow_param = self.shadow[name]
                new_param = param.data.to(self.device)

                shadow_param.mul_(self.decay).add_(new_param, alpha=(1.0 - self.decay))

    def apply_shadow(self):
        """
        Replace the model's current parameters with the shadow (EMA) parameters.
        Useful for validation or inference.
        Saves the current parameters to a backup to allow restoration.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in EMA shadow."

                # Backup current data
                self.backup[name] = param.data.clone()

                # Apply EMA data
                param.data.copy_(self.shadow[name])

        # logger.info("EMA weights applied to model.")

    def restore(self):
        """
        Restore the model's original parameters from the backup.
        Should be called after validation/inference to resume training.
        """
        if not self.backup:
            logger.warning("Attempted to restore EMA backup, but backup is empty.")
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, f"Parameter {name} not found in EMA backup."

                # Restore original data
                param.data.copy_(self.backup[name])

        # Clear backup to free memory
        self.backup = {}
        # logger.info("Original model weights restored.")

    def to(self, device):
        """
        Move shadow parameters to a specific device.

        Args:
            device (torch.device): The target device.
        """
        self.device = device
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
