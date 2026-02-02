import copy
import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.
    Wraps a timm backbone (ConvNeXt Small) with a custom configuration.
    """

    def __init__(self, model_name=None, pretrained=None):
        super(CassavaModel, self).__init__()

        # Use config defaults if not provided
        name = model_name if model_name is not None else CFG.model_name
        is_pretrained = pretrained if pretrained is not None else CFG.pretrained

        # Create the model using timm
        # drop_path_rate implements Stochastic Depth
        self.backbone = timm.create_model(
            name,
            pretrained=is_pretrained,
            num_classes=CFG.num_classes,
            drop_path_rate=CFG.drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W]
        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        return self.backbone(x)


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to improve generalization
    and stability, particularly useful for progressive resolution training.
    """

    def __init__(self, model, decay=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay rate for the moving average.
        """
        self.decay = decay if decay is not None else CFG.model_ema_decay

        # Create a deep copy of the model for the EMA weights
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        # Disable gradients for the EMA model
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA parameters using the current model parameters.
        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, param in msd.items():
                if name in esd:
                    # Apply EMA formula: shadow = decay * shadow + (1 - decay) * current
                    # Using in-place operations for efficiency
                    esd[name].copy_(self.decay * esd[name] + (1.0 - self.decay) * param)
