import copy
import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout: A regularization technique that accelerates convergence
    and improves generalization. It applies different dropout masks to the same
    features and averages the predictions from a shared linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_samples: int = 5,
        drop_rate: float = 0.5,
    ):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            num_samples (int): Number of dropout samples to average.
            drop_rate (float): Dropout probability.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(drop_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Classes).
        """
        # Apply each dropout mask and pass through the shared FC layer
        logits = [self.fc(dropout(x)) for dropout in self.dropouts]

        # Stack logits along a new dimension and compute the mean
        return torch.stack(logits, dim=0).mean(dim=0)


class CassavaClassifier(nn.Module):
    """
    Main classifier for Cassava Leaf Disease Detection.
    Uses a ConvNeXt backbone with a Multi-Sample Dropout head.
    """

    def __init__(self, config: Config, pretrained: bool = True):
        """
        Args:
            config (Config): Configuration object containing model parameters.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(CassavaClassifier, self).__init__()
        self.config = config

        # Initialize backbone using timm
        # num_classes=0 and global_pool='avg' returns the pooled feature vector
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=config.drop_path_rate,
        )

        # Determine input features for the head
        in_features = self.backbone.num_features

        # Initialize the custom classification head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=config.num_classes,
            num_samples=5,  # Standard default for MSD
            drop_rate=config.drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Classes).
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.head(features)

        return logits


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to stabilize training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for the moving average.
            device (torch.device): Device to store the EMA model on.
        """
        self.decay = decay
        # Create a deep copy of the model for the shadow weights
        self.module = copy.deepcopy(model)
        self.module.eval()

        if device is not None:
            self.module.to(device=device)

    def update(self, model: nn.Module):
        """
        Update the EMA model parameters using the current model's parameters.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters
            msd = model.state_dict()
            for name, ema_v in self.module.state_dict().items():
                if name in msd:
                    model_v = msd[name].to(ema_v.device)
                    # ema_v = decay * ema_v + (1 - decay) * model_v
                    ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)
