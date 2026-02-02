import torch
import torch.nn as nn
import timm
from copy import deepcopy
from library.config import Config


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.
    Wraps a timm backbone (ConvNeXt) with specific configuration for the task.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        pretrained: bool = True,
        num_classes: int = Config.NUM_CLASSES,
        drop_path_rate: float = Config.DROP_PATH_RATE,
    ):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of output classes.
            drop_path_rate (float): Stochastic depth rate for regularization.
        """
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.model(x)


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Includes a reset mechanism for Dynamic Fidelity Curriculum phase transitions.
    """

    def __init__(
        self, model, decay: float = Config.EMA_DECAY, device: str = Config.DEVICE
    ):
        """
        Args:
            model (nn.Module): The online model to track.
            decay (float): The decay factor for EMA (alpha).
            device (str): Device to store the shadow model on.
        """
        self.decay = decay
        self.device = device

        # Create a deep copy of the model for the shadow weights
        self.module = deepcopy(model)
        self.module.eval()
        self.module.to(self.device)

        # Disable gradients for the shadow model to save memory/compute
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Updates the shadow model weights based on the current online model.
        Formula: shadow = decay * shadow + (1 - decay) * online
        """
        with torch.no_grad():
            msd = model.state_dict()
            for name, ema_param in self.module.state_dict().items():
                model_param = msd[name]

                # Apply EMA only to floating point parameters (weights, biases)
                if ema_param.dtype in [torch.float16, torch.float32, torch.float64]:
                    ema_param.copy_(
                        ema_param * self.decay + model_param * (1.0 - self.decay)
                    )
                else:
                    # Directly copy integer buffers (e.g., num_batches_tracked)
                    ema_param.copy_(model_param)

    def reset_weights(self, model):
        """
        Hard reset of the shadow weights to match the current online model.

        This is critical for the 'Phase Reset' strategy: when switching from
        low-resolution (Phase 1) to high-resolution (Phase 2), we reset the EMA
        to prevent stale coarse features from dragging down the fine-tuning process.
        """
        self.module.load_state_dict(model.state_dict())
        self.module.eval()
