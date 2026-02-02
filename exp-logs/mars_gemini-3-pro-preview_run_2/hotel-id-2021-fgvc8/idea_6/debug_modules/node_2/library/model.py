import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the spatial features.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid division by zero or log of zero issues
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


class HotelIdModel(nn.Module):
    """
    Hotel Identification Model.
    Wraps a timm backbone, applies GeM pooling, and projects features to embedding space.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAMES[0],
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=True,
    ):
        super(HotelIdModel, self).__init__()

        # Initialize backbone
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        # instead of a pooled vector.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features from the backbone
        # by running a dummy forward pass.
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Pooling Layer
        self.pooling = GeM()

        # Neck (Projection Head)
        # BN -> Dropout -> FC -> BN
        # This structure helps in stabilizing training for ArcFace-like losses.
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)

        # Initialize weights for the neck
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Embeddings of shape (B, embedding_size)
        """
        # Feature Extraction
        x = self.backbone(x)

        # Pooling (B, C, H, W) -> (B, C, 1, 1)
        x = self.pooling(x)

        # Flatten (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1)

        # Projection
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x
