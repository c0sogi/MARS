import torch
import torch.nn as nn
import timm
from copy import deepcopy
from library.config import CFG


class CassavaModel(nn.Module):
    """
    Wrapper around timm's ConvNeXt Base model.
    Configures the classifier head for 5 classes and sets up Stochastic Depth.
    """

    def __init__(self, model_name=CFG.model_name, pretrained=True):
        super(CassavaModel, self).__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=CFG.num_classes,
            drop_path_rate=CFG.drop_path_rate,
        )

    def forward(self, x):
        return self.backbone(x)


class ModelEMA:
    """
    Implements Exponential Moving Average of model parameters.
    References:
    - https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    """

    def __init__(self, model, decay=CFG.ema_decay):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.9999).
        """
        # Create a deep copy of the model to store averaged weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay

        # Ensure EMA model is on the correct device
        self.ema.to(CFG.device)

        # Disable gradients for the EMA model to save memory/compute
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        """
        Update the EMA parameters using the current model's parameters.
        Formula: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            # Iterate over both state dicts (includes parameters and buffers like BatchNorm stats)
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for k in esd.keys():
                if k in msd:
                    model_v = msd[k].detach().to(CFG.device)
                    ema_v = esd[k]
                    # In-place update
                    ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))

    @property
    def module(self):
        """
        Returns the underlying EMA model.
        Useful for inference or saving checkpoints.
        """
        return self.ema


def get_model(pretrained=True):
    """
    Factory function to create the model and move it to the configured device.
    """
    model = CassavaModel(pretrained=pretrained)
    model.to(CFG.device)
    return model
