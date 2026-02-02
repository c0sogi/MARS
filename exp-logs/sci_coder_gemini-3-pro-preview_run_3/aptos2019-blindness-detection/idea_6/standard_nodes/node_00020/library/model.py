import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the input feature map.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    When p=1, it acts as Average Pooling.
    When p -> infinity, it acts as Max Pooling.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN gradients with negative values (though ReLU usually prevents this)
        # and ensure stability with eps
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
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


class DRModel(nn.Module):
    """
    Diabetic Retinopathy Regression Model.

    Backbone: ConvNeXt Base
    Pooling: GeM (Generalized Mean Pooling)
    Head: LayerNorm -> Dropout -> Linear (Regression)
    """

    def __init__(self, model_name="convnext_base", pretrained=True, drop_rate=0.0):
        super(DRModel, self).__init__()

        # Load backbone
        # We set num_classes=0 and global_pool='' to get the raw feature maps
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of input features for the head
        in_features = self.backbone.num_features

        # Pooling layer
        self.gem = GeM(p=3.0)

        # Regression Head
        # ConvNeXt architectures typically use LayerNorm before the final classification
        self.head = nn.Sequential(
            nn.LayerNorm(in_features), nn.Dropout(drop_rate), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # Extract features: (B, C, H, W)
        features = self.backbone(x)

        # Apply GeM pooling: (B, C, 1, 1)
        pooled = self.gem(features)

        # Flatten: (B, C)
        flat = pooled.flatten(1)

        # Regression prediction: (B, 1)
        output = self.head(flat)

        return output
