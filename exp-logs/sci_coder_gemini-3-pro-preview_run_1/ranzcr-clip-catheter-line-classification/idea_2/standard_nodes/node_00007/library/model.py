import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small value to avoid numerical instability.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN gradients with negative inputs (though usually relu precedes this)
        # or zeros when p < 1.
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


class CatheterModel(nn.Module):
    """
    Catheter Detection Model based on EfficientNet-B4 with GeM Pooling.

    Architecture:
    1. Backbone: tf_efficientnet_b4 (pretrained on ImageNet)
    2. Pooling: Generalized Mean Pooling (GeM)
    3. Head: Dropout + Linear Layer
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super().__init__()

        # Create backbone using timm
        # features_only=False allows us to use reset_classifier logic easily,
        # but we use forward_features method which works on standard models.
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Get the number of features output by the backbone
        # timm models usually have a num_features attribute
        self.in_features = self.backbone.num_features

        # Remove the original classification head
        # This is strictly not necessary if we use forward_features, but cleaner
        self.backbone.reset_classifier(0)

        # Pooling layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Dropout layer
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)

        # Classification head
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES)
        """
        # Extract features from backbone
        # Shape: (B, C, H', W')
        features = self.backbone.forward_features(x)

        # Apply pooling
        # Shape: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten
        # Shape: (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Apply dropout
        features_dropped = self.dropout(flattened_features)

        # Classification
        # Shape: (B, NUM_CLASSES)
        logits = self.fc(features_dropped)

        return logits
