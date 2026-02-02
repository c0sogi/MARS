import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter, initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp input to eps to avoid numerical instability with pow
        x = x.clamp(min=eps)

        # Calculate x^p
        x_pow = x.pow(p)

        # Average pooling over spatial dimensions (H, W)
        # Result shape: (Batch, Channels, 1, 1)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        return avg_pool.pow(1.0 / p)

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


class RetinopathyModel(nn.Module):
    """
    Retinopathy Severity Prediction Model.
    Consists of:
    1. A backbone CNN/Transformer (from timm) to extract spatial features.
    2. A GeM pooling layer to aggregate features, focusing on high activations.
    3. A linear regression head to predict the severity score.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model (e.g., 'tf_efficientnet_b5_ns').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(RetinopathyModel, self).__init__()

        # Create backbone model
        # num_classes=0 and global_pool='' ensures we get the raw feature maps
        # (B, C, H, W) instead of a pooled vector or class logits.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Dynamically determine the number of input channels for the head
        # by running a dummy forward pass. This makes the class compatible
        # with any backbone supported by timm.
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.image_size, Config.image_size)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Generalized Mean Pooling
        self.pool = GeM()

        # Regression Head: Maps flattened features to a single scalar
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Predicted severity scores of shape (B, 1)
        """
        # Feature Extraction -> (B, C, H_feat, W_feat)
        features = self.backbone(x)

        # GeM Pooling -> (B, C, 1, 1)
        pooled = self.pool(features)

        # Flatten -> (B, C)
        flattened = torch.flatten(pooled, 1)

        # Regression -> (B, 1)
        output = self.head(flattened)

        return output
