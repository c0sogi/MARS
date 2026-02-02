import copy
import torch
from library.config import Config


class ModelEMA:
    """
    Model Exponential Moving Average (EMA) utility.
    Maintains a shadow copy of the model weights that are updated using an exponential decay.
    This helps in stabilizing training and often leads to better generalization.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=Config.DEVICE):
        """
        Initialize the EMA model.

        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay rate for EMA (default: from Config).
            device (torch.device): The device to store the EMA model on.
        """
        self.decay = decay
        self.device = device

        # Create a deep copy of the model to serve as the shadow model
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.module.to(self.device)

        # Ensure the shadow model does not require gradients
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters based on the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            # Get the state dictionary of the current model
            # This handles matching parameters by name
            msd = model.state_dict()

            # Update parameters using the EMA formula:
            # shadow_param = decay * shadow_param + (1 - decay) * current_param
            for name, param in self.module.named_parameters():
                if name in msd:
                    # Perform in-place update for efficiency
                    # .to(self.device) ensures tensor is on the correct device before operation
                    param.mul_(self.decay).add_(
                        msd[name].to(self.device), alpha=1 - self.decay
                    )

            # Update buffers (e.g., BatchNorm running mean/var)
            # Buffers are typically copied directly rather than averaged to track
            # the most recent statistics accurately.
            for name, buffer in self.module.named_buffers():
                if name in msd:
                    buffer.copy_(msd[name].to(self.device))
