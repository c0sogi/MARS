import torch
import torch.nn as nn
import timm
from copy import deepcopy
from library.config import Config


def get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Factory function to create models using the timm library.

    Args:
        model_name (str): Name of the model architecture in timm.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to create model {model_name}: {e}")


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a moving average of model weights to improve stability and performance.
    Useful for stabilizing training of Transformers and improving final model robustness.
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for the moving average (default: 0.9999).
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.ema = deepcopy(model)
        self.ema.eval()

        # Disable gradients for the shadow model to save memory/compute
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.

        Args:
            model (nn.Module): The current training model.
        """
        # Unwrap DataParallel/DistributedDataParallel if necessary
        if hasattr(model, "module"):
            model = model.module

        with torch.no_grad():
            # Iterate over state_dict to handle both parameters and buffers (e.g., BN stats)
            # Using zip assumes state_dicts are ordered and identical in structure
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, m_tensor in msd.items():
                if name in esd:
                    e_tensor = esd[name]

                    if m_tensor.dtype.is_floating_point:
                        # Apply EMA update: shadow = decay * shadow + (1 - decay) * current
                        e_tensor.copy_(
                            e_tensor * self.decay + m_tensor * (1.0 - self.decay)
                        )
                    else:
                        # Directly copy non-floating point tensors (e.g., integer buffers like num_batches_tracked)
                        e_tensor.copy_(m_tensor)
