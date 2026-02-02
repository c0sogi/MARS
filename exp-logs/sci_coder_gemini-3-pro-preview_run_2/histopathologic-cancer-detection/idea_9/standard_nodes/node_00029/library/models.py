import copy
import torch
import torch.nn as nn
import timm
from library.config import Config


class PathologyClassifier(nn.Module):
    """
    A wrapper around timm models to create a heterogeneous ensemble component.

    This class ensures:
    1. Usage of Global Average Pooling (GAP) via global_pool='avg'.
    2. Retention of backbone-specific normalization layers (LayerNorm for ConvNeXt,
       BatchNorm for EfficientNet) immediately preceding the classifier.
    3. Proper initialization for binary classification (num_classes=1).
    """

    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(PathologyClassifier, self).__init__()
        self.model_name = model_name

        # Create the model using timm.
        # By specifying num_classes and global_pool='avg', timm automatically
        # configures the head to include the specific normalization layer
        # required by the backbone (e.g., LayerNorm for ConvNeXt) before the FC layer.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool="avg",
        )

    def forward(self, x):
        return self.model(x)


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) of model parameters.

    EMA maintains a moving average of the model weights, which often leads to
    better generalization and stability, especially when using short training
    schedules (e.g., 25 epochs) where the final weights might still be noisy.
    """

    def __init__(self, model, decay=None):
        """
        Args:
            model: The source model to track.
            decay: The decay factor (beta). If None, defaults to Config.EMA_DECAY.
        """
        self.decay = decay if decay is not None else Config.EMA_DECAY

        # Create a deep copy of the model to serve as the shadow model
        self.model = copy.deepcopy(model)

        # Set the shadow model to evaluation mode
        self.model.eval()

        # Disable gradients for the shadow model to save memory/compute
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.
        Formula: shadow_param = decay * shadow_param + (1 - decay) * current_param
        """
        with torch.no_grad():
            # Update parameters
            for p_ema, p_model in zip(self.model.parameters(), model.parameters()):
                # We update in-place to save memory
                p_ema.data.mul_(self.decay).add_(p_model.data, alpha=1 - self.decay)

            # Update buffers (e.g., BatchNorm running mean/var)
            # We strictly copy buffers from the current model to the shadow model
            for b_ema, b_model in zip(self.model.buffers(), model.buffers()):
                b_ema.copy_(b_model)

    def to(self, device):
        """Moves the shadow model to the specified device."""
        self.model.to(device)
        return self
