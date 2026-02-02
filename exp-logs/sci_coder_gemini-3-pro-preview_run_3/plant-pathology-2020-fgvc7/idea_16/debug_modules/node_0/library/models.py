import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the spatial dimensions of the input tensor.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp for numerical stability
        x = x.clamp(min=eps).pow(p)

        # Average pooling over spatial dimensions (H, W)
        # kernel_size matches the spatial dimensions of the input
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Root p
        x = x.pow(1.0 / p)
        return x

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
    Apple Disease Classification Model.

    Architecture:
    1. Backbone (EfficientNetV2 or Swin) initialized with features_only=True.
    2. Multi-Level GeM Pooling: Applies GeM to the last 3 feature maps.
    3. Concatenation: Joins the pooled features.
    4. Linear Head: Maps concatenated features to class logits.
    """

    def __init__(self, backbone_name, num_classes, pretrained=True):
        super(AppleNet, self).__init__()

        # Initialize backbone using timm
        # We pass img_size to ensure Transformers (like Swin) initialize
        # position embeddings/window sizes correctly for the target resolution.
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                img_size=Config.IMG_SIZE,
            )
        except TypeError:
            # Fallback for models that might not accept img_size in create_model
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, features_only=True
            )

        # Perform a dummy forward pass to determine feature map channels
        # This makes the model agnostic to the specific backbone's output structure
        dummy_input = torch.randn(2, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        with torch.no_grad():
            features = self.backbone(dummy_input)

        # We select the last 3 feature maps (strides 8, 16, 32 typically)
        # features is a list of tensors. We take the last 3.
        if len(features) < 3:
            raise ValueError(
                f"Backbone {backbone_name} returns fewer than 3 feature maps."
            )

        selected_features = features[-3:]
        self.feature_channels = [f.shape[1] for f in selected_features]

        # Create a ModuleList of GeM poolers, one for each selected feature level
        self.poolers = nn.ModuleList(
            [GeM(p=Config.GEM_P) for _ in range(len(self.feature_channels))]
        )

        # Calculate total feature dimension after concatenation
        total_features = sum(self.feature_channels)

        # Simple Linear Head (No Dropout/BN as per instructions)
        self.fc = nn.Linear(total_features, num_classes)

    def forward(self, x):
        # Extract features from backbone
        # Returns a list of tensors
        features = self.backbone(x)

        # Select the last 3 feature maps
        selected_features = features[-3:]

        pooled_features = []
        for i, feat in enumerate(selected_features):
            # Apply GeM pooling: (B, C, H, W) -> (B, C, 1, 1)
            p = self.poolers[i](feat)

            # Flatten: (B, C, 1, 1) -> (B, C)
            p = torch.flatten(p, 1)
            pooled_features.append(p)

        # Concatenate features from all levels: (B, Sum(C))
        concat = torch.cat(pooled_features, dim=1)

        # Classification
        logits = self.fc(concat)

        return logits
