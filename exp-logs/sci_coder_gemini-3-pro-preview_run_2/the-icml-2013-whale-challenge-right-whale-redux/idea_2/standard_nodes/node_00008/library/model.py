import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp input for numerical stability
        x = x.clamp(min=eps)

        # Raise to power p
        x = x.pow(p)

        # Average pooling over spatial dimensions
        # Output shape: (Batch, Channels, 1, 1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", eps="
            + str(self.eps)
            + ")"
        )


class WhaleEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model for Right Whale Detection.
    Features:
    - 1-channel input adaptation (weights averaged from RGB)
    - Generalized Mean (GeM) Pooling
    - Binary classification head
    """

    def __init__(self, config, pretrained=True):
        super(WhaleEfficientNet, self).__init__()
        self.config = config

        # Load EfficientNet-B0 backbone from timm
        # num_classes=0 removes the default classifier
        # global_pool='' removes the default pooling, returning feature maps
        self.backbone = timm.create_model(
            config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Modify the first convolutional layer to accept 1-channel input
        if config.IN_CHANNELS != 3:
            # EfficientNet uses 'conv_stem' as the first layer
            old_stem = self.backbone.conv_stem
            new_stem = nn.Conv2d(
                in_channels=config.IN_CHANNELS,
                out_channels=old_stem.out_channels,
                kernel_size=old_stem.kernel_size,
                stride=old_stem.stride,
                padding=old_stem.padding,
                bias=(old_stem.bias is not None),
            )

            # Initialize weights by averaging the original RGB weights
            # old_stem.weight shape: (Out, 3, K, K) -> new_stem.weight shape: (Out, 1, K, K)
            with torch.no_grad():
                new_stem.weight.copy_(old_stem.weight.mean(dim=1, keepdim=True))
                if old_stem.bias is not None:
                    new_stem.bias.copy_(old_stem.bias)

            self.backbone.conv_stem = new_stem

        # Pooling Layer
        if config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # backbone.num_features gives the number of output channels from the backbone
        self.in_features = self.backbone.num_features
        self.fc = nn.Linear(self.in_features, config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, Freq, Time)
        Returns:
            logits (torch.Tensor): Raw output scores of shape (Batch, 1)
        """
        # Extract features from backbone
        # Shape: (Batch, Channels, F', T')
        features = self.backbone(x)

        # Apply Pooling
        # Shape: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # Flatten
        # Shape: (Batch, Channels)
        flattened = pooled.view(pooled.size(0), -1)

        # Classification
        # Shape: (Batch, 1)
        logits = self.fc(flattened)

        return logits
