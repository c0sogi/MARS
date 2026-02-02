import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.

    Learns a pooling parameter 'p' to interpolate between Max Pooling (p -> infinity)
    and Average Pooling (p -> 1). This is effective for fine-grained visual recognition.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp for numerical stability before power operation
        x = x.clamp(min=eps).pow(p)

        # Global Average Pooling on the power-transformed feature map
        # Kernel size is set to the spatial dimensions of the input (H, W)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Inverse power to return to original scale
        return x.pow(1.0 / p)

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


class WhaleModel(nn.Module):
    """
    Whale Identification Model.

    Architecture:
    1. Backbone: EfficientNet-B4 (pretrained, with Gradient Checkpointing)
    2. Pooling: Generalized Mean (GeM) Pooling
    3. Head: Dropout -> Linear -> BatchNorm

    Outputs:
    - Embeddings (batch_size, embedding_size)
    """

    def __init__(
        self, embedding_size=Config.embedding_size, pretrained=Config.pretrained
    ):
        super(WhaleModel, self).__init__()

        # ---------------------------------------------------------
        # 1. Backbone
        # ---------------------------------------------------------
        # We use timm to create the backbone.
        # num_classes=0 removes the top classification layer.
        # global_pool="" removes the default pooling, keeping (B, C, H, W) output.
        self.backbone = timm.create_model(
            Config.backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # ---------------------------------------------------------
        # 2. Gradient Checkpointing
        # ---------------------------------------------------------
        # Critical for memory efficiency with B4 @ 384px.
        # This trades compute (re-calculating activations) for memory.
        if Config.use_gradient_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        # Get the number of output features from the backbone
        self.in_features = self.backbone.num_features

        # ---------------------------------------------------------
        # 3. Pooling & Head
        # ---------------------------------------------------------
        self.pooling = GeM()

        self.dropout = nn.Dropout(p=Config.dropout_rate)
        self.fc = nn.Linear(self.in_features, embedding_size)

        # BatchNorm is often used after the linear layer in ArcFace/CosFace setups
        # to normalize the scale of features before the angular margin loss.
        self.bn = nn.BatchNorm1d(embedding_size)

        # Initialize Head Weights
        self._init_params()

    def _init_params(self):
        """
        Initialize the weights of the embedding head.
        """
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature embeddings of shape (B, embedding_size).
        """
        # 1. Backbone Feature Extraction
        # Output shape: (B, C, H', W')
        features = self.backbone(x)

        # 2. GeM Pooling
        # Output shape: (B, C, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Output shape: (B, C)
        flattened = pooled.view(pooled.size(0), -1)

        # 4. Embedding Head
        x = self.dropout(flattened)
        x = self.fc(x)
        embeddings = self.bn(x)

        return embeddings
