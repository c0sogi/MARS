import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    This layer computes the generalized mean of the spatial features, which allows
    the model to focus on salient regions (like scars or edges) while suppressing
    background noise. It is a learnable pooling operation where 'p' is a trainable parameter.

    Formula: f(X) = (1/N * sum(x^p))^(1/p)

    Reference: Radenovic et al. "Fine-tuning CNN Image Retrieval with No Annotation"
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is initialized to 3.0 and is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)

        # Clamp inputs to avoid numerical instability (NaNs) when raising to power p
        # especially if the backbone uses activation functions that allow negative values (though rare at end)
        x = x.clamp(min=self.eps)

        # Apply Average Pooling on x^p over the spatial dimensions (H, W)
        # Then raise the result to the power of 1/p
        # Result shape: (Batch, Channels, 1, 1)
        return F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleModel(nn.Module):
    """
    Whale Identification Model.

    This class wraps a `timm` backbone (EfficientNet-B4 or V2-M) and replaces the
    default head with a GeM pooling layer and a linear projection layer.

    It outputs raw embeddings which are intended to be normalized and processed
    by the ArcFace loss function during training, or used for cosine similarity
    search during inference.
    """

    def __init__(self, model_name, pretrained=True, embedding_size=512):
        """
        Args:
            model_name (str): The timm model identifier (e.g., 'tf_efficientnet_b4').
            pretrained (bool): Whether to load ImageNet pretrained weights.
            embedding_size (int): The dimensionality of the output embedding vector.
        """
        super(WhaleModel, self).__init__()

        # Create the backbone using timm
        # num_classes=0 removes the default classification layer
        # global_pool='' removes the default pooling layer, returning spatial feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Enable Gradient Checkpointing if configured
        # This trades compute for memory, allowing larger batch sizes or resolutions
        if Config.USE_GRADIENT_CHECKPOINTING:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(True)

        # Determine the number of input channels for the head
        # timm models expose this via num_features
        self.in_features = self.backbone.num_features

        # Define the custom head
        self.pooling = GeM()
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.in_features, embedding_size)

        # Initialize weights for the new fully connected layer
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Feature embeddings of shape (Batch, Embedding_Size).
        """
        # 1. Feature Extraction
        # Output shape: (Batch, Channels, Height, Width)
        features = self.backbone(x)

        # 2. GeM Pooling
        # Output shape: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Output shape: (Batch, Channels)
        flattened = pooled.view(pooled.size(0), -1)

        # 4. Dropout
        dropped = self.dropout(flattened)

        # 5. Projection to Embedding Space
        # Output shape: (Batch, Embedding_Size)
        embeddings = self.fc(dropped)

        return embeddings
