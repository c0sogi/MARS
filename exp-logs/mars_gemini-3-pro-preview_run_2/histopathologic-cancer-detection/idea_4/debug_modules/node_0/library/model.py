import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (AvgPool(x^p))^(1/p).

    When p=1, it acts as Average Pooling.
    When p->infinity, it acts as Max Pooling.
    The parameter p is learnable.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # 1. Clamp to avoid numerical instability with pow
        # 2. Raise to power p
        # 3. Average pool over the spatial dimensions
        # 4. Raise to power 1/p
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


class TumorClassifier(nn.Module):
    """
    Binary classifier for tumor detection using a ConvNeXt backbone and GeM pooling.
    """

    def __init__(self, model_name=None, pretrained=None):
        """
        Args:
            model_name (str, optional): Name of the timm backbone. Defaults to Config.model_name.
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to Config.pretrained.
        """
        super(TumorClassifier, self).__init__()

        # Use defaults from Config if arguments are not provided
        model_name = model_name or Config.model_name
        pretrained = pretrained if pretrained is not None else Config.pretrained

        # Load the backbone from timm
        # num_classes=0 and global_pool='' ensures we get the spatial feature maps (B, C, H, W)
        # instead of a pooled vector or logits.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        in_features = self.backbone.num_features

        # Initialize Pooling Layer
        if Config.use_gem_pooling:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize Classification Head
        # Output is 1 logit for binary classification (BCEWithLogitsLoss)
        self.fc = nn.Linear(in_features, Config.num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # 1. Feature Extraction
        features = self.backbone(x)  # Shape: (B, C, H_feat, W_feat)

        # 2. Pooling
        pooled_features = self.pool(features)  # Shape: (B, C, 1, 1)

        # 3. Flatten
        flattened_features = torch.flatten(pooled_features, 1)  # Shape: (B, C)

        # 4. Classification
        logits = self.fc(flattened_features)  # Shape: (B, 1)

        return logits
