import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean of the input tensor. When p=1, it is equivalent
    to Average Pooling. When p -> infinity, it approaches Max Pooling.
    The parameter p is learnable.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter, initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Pooled tensor of shape (Batch, Channels, 1, 1).
        """
        # Clamp to avoid numerical instability with pow
        x = x.clamp(min=self.eps)

        # Average pooling on x^p
        x_pow = x.pow(self.p)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        return avg_pool.pow(1.0 / self.p)


class WhaleEfficientNet(nn.Module):
    """
    EfficientNetV2-based model for Right Whale Detection.

    Uses a pretrained EfficientNetV2 backbone, replaces the global pooling
    with GeM pooling, and adds a linear classification head.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super(WhaleEfficientNet, self).__init__()

        # Create backbone with no classification head and no global pooling
        # We use 'tf_efficientnetv2_m' as defined in Config
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine the number of input features for the linear layer dynamically.
        # We create a dummy input matching the spectrogram shape: (Batch, 3, N_MELS, Time)
        # Time dimension is calculated based on N_SAMPLES and HOP_LENGTH.
        time_steps = Config.N_SAMPLES // Config.HOP_LENGTH + 1
        dummy_input = torch.randn(1, 3, Config.N_MELS, time_steps)

        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Generalized Mean Pooling
        self.pooling = GeM()

        # Classification Head
        self.dropout = nn.Dropout(Config.DROP_RATE)
        # Output is 1 logit for binary classification (BCEWithLogitsLoss)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 3, Freq, Time).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Extract features from backbone
        x = self.backbone(x)

        # Apply GeM pooling
        x = self.pooling(x)

        # Flatten (Batch, Channels, 1, 1) -> (Batch, Channels)
        x = x.flatten(1)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x
