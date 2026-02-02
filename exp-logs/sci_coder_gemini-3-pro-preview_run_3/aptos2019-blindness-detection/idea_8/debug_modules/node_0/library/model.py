import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (batch_size, channels, height, width)
        # Clamp to avoid numerical instability with negative values or zeros before power
        x = x.clamp(min=eps)
        # Apply formula: (Avg(x^p))^(1/p)
        # We use avg_pool2d to calculate the mean over the spatial dimensions (H, W)
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


class DRModel(nn.Module):
    """
    Diabetic Retinopathy Classification Model.
    Architecture: ConvNeXt Backbone -> GeM Pooling -> Linear Regression Head
    """

    def __init__(self, model_name=None, pretrained=None):
        super(DRModel, self).__init__()

        # Load configuration defaults if not provided
        if model_name is None:
            model_name = Config.model_name
        if pretrained is None:
            pretrained = Config.pretrained

        # Create Backbone using timm
        # num_classes=0 and global_pool="" ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=Config.drop_rate,
            drop_path_rate=Config.drop_path_rate,
        )

        # Determine the number of input features for the head
        in_features = self.backbone.num_features

        # Pooling Layer
        self.gem = GeM()

        # Regression Head
        # Output is 1 neuron for regression (severity score)
        self.head = nn.Linear(in_features, Config.num_classes)

    def forward(self, x):
        # 1. Extract features from backbone
        # Shape: (Batch, Channels, Height, Width)
        features = self.backbone(x)

        # 2. Apply Generalized Mean Pooling
        # Shape: (Batch, Channels, 1, 1)
        pooled_features = self.gem(features)

        # 3. Flatten
        # Shape: (Batch, Channels)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # 4. Regression Output
        # Shape: (Batch, 1)
        output = self.head(flattened_features)

        return output
