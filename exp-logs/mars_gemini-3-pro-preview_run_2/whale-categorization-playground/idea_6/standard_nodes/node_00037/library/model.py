import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes the generalized mean of each channel in the feature map.
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
        # Clamp to avoid NaN with power
        x = x.clamp(min=eps)
        # Average pooling on x^p
        x_pow = x.pow(p)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # Raise to power 1/p
        return avg_pool.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleModel(nn.Module):
    """
    Whale Species Prediction Model.

    Architecture:
    1. Backbone: EfficientNet-B4 (initialized via timm)
    2. Pooling: Generalized Mean Pooling (GeM)
    3. Neck: Linear Projection -> BatchNorm

    The model outputs an embedding vector. The classification head (ArcFace)
    is handled by the loss function during training.
    """

    def __init__(self, embedding_size=Config.EMBEDDING_SIZE, pretrained=True):
        super(WhaleModel, self).__init__()

        # Initialize Backbone
        # num_classes=0 removes the default classifier
        # global_pool="" removes the default pooling, returning (B, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        # Run a dummy forward pass to get shapes
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Pooling Layer
        self.pooling = GeM()

        # Neck (Projection Head)
        # Projects backbone features to the embedding space
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
            # Note: No activation here (e.g., ReLU), as we want raw embeddings for ArcFace
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images, shape (B, 3, H, W)

        Returns:
            torch.Tensor: Feature embeddings, shape (B, embedding_size)
        """
        # 1. Backbone Feature Extraction
        x = self.backbone(x)  # Shape: (B, C, H_feat, W_feat)

        # 2. GeM Pooling
        x = self.pooling(x)  # Shape: (B, C, 1, 1)

        # 3. Flatten
        x = x.flatten(1)  # Shape: (B, C)

        # 4. Projection Neck
        embeddings = self.neck(x)  # Shape: (B, embedding_size)

        return embeddings

    def enable_gradient_checkpointing(self):
        """
        Enables gradient checkpointing (activation checkpointing) on the backbone.
        This trades compute for memory, allowing larger batch sizes or larger models (like B4).
        """
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        else:
            print(
                f"Warning: Backbone {Config.BACKBONE} does not support set_grad_checkpointing via timm."
            )
