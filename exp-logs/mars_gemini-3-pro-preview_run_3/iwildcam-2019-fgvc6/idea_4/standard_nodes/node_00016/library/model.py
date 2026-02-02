import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor over the spatial dimensions.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN with negative values (if any) or zero
        # ConvNeXt uses GELU, so values can be negative.
        # GeM is typically applied to ReLU outputs, but clamping handles general cases.
        # We clamp min to eps to ensure stability for pow operation.
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


class AnimalModel(nn.Module):
    """
    Animal Species Classification Model.

    Architecture:
    - Backbone: ConvNeXt V2 Tiny (pretrained)
    - Pooling: Generalized Mean Pooling (GeM)
    - Head: Multi-Sample Dropout + Linear Layer
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(AnimalModel, self).__init__()

        # Load backbone using timm
        # global_pool='' ensures we get spatial features (B, C, H, W)
        # num_classes=0 removes the default classification head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features for the head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback inspection if num_features is not explicitly set
            # Pass a dummy input to check output shape
            with torch.no_grad():
                dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Pooling layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Multi-Sample Dropout
        # We create multiple dropout layers with rates defined in Config
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.DROPOUT_RATES])

        # Final Classification Layer
        self.fc = nn.Linear(in_features, num_classes)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Extract features from backbone
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # Apply pooling
        # Shape: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten
        # Shape: (B, C)
        flattened_features = torch.flatten(pooled_features, 1)

        # Multi-Sample Dropout Head
        # Apply each dropout mask, pass through FC, and average the results
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.fc(dropout(flattened_features))
            else:
                output += self.fc(dropout(flattened_features))

        # Average the predictions
        output /= len(self.dropouts)

        return output
