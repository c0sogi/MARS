import torch
import torch.nn as nn
import timm
from copy import deepcopy
from library.config import Config


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.
    Wraps a timm backbone (ConvNeXt Small) with a custom head configuration.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        drop_path_rate=Config.DROP_PATH_RATE,
    ):
        super().__init__()

        # Create the backbone model using timm
        # drop_path_rate implements Stochastic Depth
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.backbone(x)


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters for better generalization.
    Includes a 'reset_weights' feature for Curriculum Learning phase transitions.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=None):
        """
        Args:
            model: The model to track.
            decay: The decay factor for EMA (beta).
            device: Device to store the shadow model on.
        """
        self.decay = decay
        self.device = device if device else Config.DEVICE

        # Create a deep copy of the model to serve as the shadow model
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.shadow.to(self.device)

        # Freeze all parameters in the shadow model
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.
        formula: shadow_param = decay * shadow_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            # Update parameters
            # We zip to iterate over both models simultaneously.
            # deepcopy guarantees the order is preserved.
            for (name, param), (shadow_name, shadow_param) in zip(
                model.named_parameters(), self.shadow.named_parameters()
            ):
                if param.requires_grad:
                    # Update shadow param: beta * shadow + (1-beta) * current
                    shadow_param.data.mul_(self.decay).add_(
                        param.data, alpha=(1.0 - self.decay)
                    )

            # Update buffers (e.g., BatchNorm running mean/var)
            # Standard practice is to copy buffers from the training model
            for (name, buffer), (shadow_name, shadow_buffer) in zip(
                model.named_buffers(), self.shadow.named_buffers()
            ):
                shadow_buffer.copy_(buffer)

    def reset_weights(self, model):
        """
        Phase Reset Strategy:
        Resets the shadow model weights to match the current state of the provided model.
        This is used when switching training phases (e.g., changing resolution) to
        ensure the EMA model doesn't lag with features learned at the previous resolution.
        """
        # Load state dict copies both parameters and buffers
        self.shadow.load_state_dict(model.state_dict())
        self.shadow.eval()  # Ensure it remains in eval mode
        # No need to set requires_grad=False again as load_state_dict doesn't change that attribute


def create_model(pretrained=True):
    """
    Factory function to create the CassavaModel based on Config.
    """
    model = CassavaModel(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=pretrained,
        drop_path_rate=Config.DROP_PATH_RATE,
    )
    return model
