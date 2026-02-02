import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    When p=1, it acts as Average Pooling.
    When p->infinity, it acts as Max Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Apply pooling over spatial dimensions (Height, Width)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class WhaleModel(nn.Module):
    """
    Right Whale Detection Model.
    Backbone: EfficientNetV2-Medium
    Pooling: GeM
    Head: Dropout + Linear
    """

    def __init__(self, config=Config, pretrained=True):
        super(WhaleModel, self).__init__()
        self.config = config

        # Initialize backbone using timm
        # in_chans=1: Adapt first conv layer for single-channel spectrograms
        # num_classes=0, global_pool="": Remove default head and pooling
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # Get the number of output features from the backbone
        self.in_features = self.backbone.num_features

        # Pooling Layer
        if config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # Includes Dropout for regularization as per Config
        self.drop = nn.Dropout(p=config.DROPOUT_RATE)
        self.fc = nn.Linear(self.in_features, config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, Freq, Time)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Feature Extraction
        # Output shape: (Batch, Channels, H_feat, W_feat)
        x = self.backbone(x)

        # Pooling
        # Output shape: (Batch, Channels, 1, 1)
        x = self.pooling(x)

        # Flatten
        # Output shape: (Batch, Channels)
        x = x.flatten(1)

        # Dropout
        x = self.drop(x)

        # Classification
        # Output shape: (Batch, Num_Classes)
        x = self.fc(x)

        return x
