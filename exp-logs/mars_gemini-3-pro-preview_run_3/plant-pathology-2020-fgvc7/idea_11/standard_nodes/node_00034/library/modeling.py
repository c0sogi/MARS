import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the spatial dimensions of the input feature map.
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
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN with negative values (though usually ReLU precedes this)
        # Apply average pooling to x^p
        # Raise result to 1/p
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


class AppleNet(nn.Module):
    """
    Apple Disease Detection Model.
    Uses a timm backbone (EfficientNetV2 or MaxViT) to extract multi-scale features,
    applies GeM pooling, and concatenates them for classification.
    """

    def __init__(self, model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model to use as backbone.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(AppleNet, self).__init__()

        # Initialize backbone in features_only mode
        # We extract the last three reduction stages (indices 2, 3, 4)
        # These correspond to the deepest layers with rich semantic information
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Get channel counts for the extracted feature maps
        # feature_info.channels() returns a list of integers
        self.feature_channels = self.backbone.feature_info.channels()

        # GeM Pooling Layer
        # We use a single GeM instance (shared p) for simplicity,
        # effectively learning a global pooling strategy for the network.
        self.gem = GeM()

        # Calculate total input dimension for the final linear layer
        # Sum of channels from the 3 extracted stages
        self.in_features = sum(self.feature_channels)

        # Final Classification Head
        self.head = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
        Returns:
            torch.Tensor: Logits (B, Num_Classes)
        """
        # Extract features from backbone
        # Returns a list of tensors corresponding to out_indices
        features = self.backbone(x)

        pooled_features = []
        for f in features:
            # f shape: (Batch, Channels, H_i, W_i)
            # Apply GeM pooling -> (Batch, Channels, 1, 1)
            pooled = self.gem(f)
            # Flatten to vector -> (Batch, Channels)
            pooled = pooled.flatten(1)
            pooled_features.append(pooled)

        # Concatenate feature vectors from all stages
        # Shape: (Batch, Sum(Channels))
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification
        logits = self.head(concat_features)

        return logits
