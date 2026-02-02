import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes (1/N * sum(x^p))^(1/p) where p is a learnable parameter.
    This pooling strategy is effective for fine-grained classification tasks
    as it focuses on salient features while suppressing background noise.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is initialized to 3.0 and is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp inputs to eps to avoid numerical instability with pow() on negative or zero values.
        # This implicitly treats negative activations (if any) as negligible/background.
        x = x.clamp(min=self.eps)

        # Calculate x^p
        x_pow = x.pow(self.p)

        # Average pooling over the spatial dimensions (H, W)
        # kernel_size is set to the spatial size of the input feature map
        pooled = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to the power of 1/p to return to original scale
        x_gem = pooled.pow(1.0 / self.p)

        return x_gem


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model using ConvNeXt-Small backbone and GeM Pooling.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(AppleDiseaseModel, self).__init__()

        # Load the backbone from timm
        # num_classes=0 and global_pool="" ensures we get the spatial feature maps (B, C, H, W)
        # instead of a pooled vector or logits. This is required for GeM pooling.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of output channels from the backbone
        self.in_features = self.backbone.num_features

        # Initialize Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        # Maps the feature vector to the number of disease classes
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Extract spatial features from backbone
        # Shape: (Batch, Channels, H_feat, W_feat)
        features = self.backbone(x)

        # Apply pooling to get global features
        # Shape: (Batch, Channels, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten the features
        # Shape: (Batch, Channels)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Apply classification head
        # Shape: (Batch, Num_Classes)
        logits = self.fc(flattened_features)

        return logits
