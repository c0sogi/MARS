import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability with pow
        x = torch.clamp(x, min=eps)
        # Apply average pooling on x^p, then take the (1/p)-th root
        # Pooling kernel size matches the spatial dimensions (H, W)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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


class WhaleClassifier(nn.Module):
    """
    Binary classifier for Whale Detection using a ConvNeXt backbone and GeM pooling.
    """

    def __init__(self):
        super(WhaleClassifier, self).__init__()

        # Initialize Backbone
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        # num_classes=0 removes the default classification head
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANS,
            num_classes=0,
            global_pool="",
        )

        # Retrieve the number of output features from the backbone
        self.in_features = self.backbone.num_features

        # Pooling Layer
        if Config.USE_GEM:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # Maps backbone features to a single logit (binary classification)
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input spectrograms of shape (B, 1, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Feature Extraction
        features = self.backbone(x)  # Output: (B, C, H_feat, W_feat)

        # 2. Pooling
        pooled = self.pooling(features)  # Output: (B, C, 1, 1)

        # 3. Flatten
        flattened = pooled.view(pooled.size(0), -1)  # Output: (B, C)

        # 4. Classification
        logits = self.fc(flattened)  # Output: (B, 1)

        return logits
