import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor.
    f(X) = (1/|X| * sum(x^p))^(1/p)

    When p=1, it acts as Average Pooling.
    When p -> infinity, it acts as Max Pooling.
    p is a learnable parameter.
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
        # Apply spatial pooling over Height and Width
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


class ArtworkModel(nn.Module):
    """
    Main model architecture for Artwork Attribute Labeling.
    Uses a ConvNeXt backbone, GeM pooling, and a linear classification head.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Args:
            model_name (str): Name of the timm model backbone.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(ArtworkModel, self).__init__()

        # Load the backbone model
        # num_classes=0 removes the default classification head
        # global_pool="" removes the default pooling layer, keeping spatial features
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: Forward a dummy input to inspect output channels
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

        # Generalized Mean Pooling
        self.gem = GeM()

        # Classification Head
        self.head = nn.Linear(in_features, num_classes)

        # Initialize weights for the head
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images, shape (B, 3, H, W).

        Returns:
            torch.Tensor: Raw logits, shape (B, num_classes).
        """
        # 1. Backbone Feature Extraction
        # Output shape: (B, C, H_feat, W_feat)
        features = self.backbone(x)

        # 2. GeM Pooling
        # Output shape: (B, C, 1, 1)
        pooled_features = self.gem(features)

        # 3. Flatten
        # Output shape: (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # 4. Classification Head
        # Output shape: (B, num_classes)
        logits = self.head(flattened_features)

        return logits
