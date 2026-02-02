import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean of each channel in the feature map:
    f = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial power value.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp input to eps to ensure numerical stability during power operation
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", eps="
            + str(self.eps)
            + ")"
        )


class CatheterModel(nn.Module):
    """
    Catheter Detection Model architecture.

    Backbone: ConvNeXt-Small (pretrained)
    Pooling: Generalized Mean Pooling (GeM)
    Head: LayerNorm -> Dropout -> Linear
    """

    def __init__(
        self,
        model_name=Config.model_name,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
        drop_rate=Config.drop_rate,
        drop_path_rate=Config.drop_path_rate,
    ):
        super().__init__()

        # Load the backbone from timm
        # global_pool='' and num_classes=0 return the unpooled feature map (B, C, H, W)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )

        # Retrieve the number of output channels from the backbone
        self.in_features = self.backbone.num_features

        # Initialize Pooling Layer
        if Config.use_gem_pooling:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Initialize Classification Head
        # ConvNeXt architectures typically benefit from LayerNorm before the final classifier
        self.norm = nn.LayerNorm(self.in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.in_features, num_classes)

        # Initialize weights for the new head layers
        self._init_head_weights()

    def _init_head_weights(self):
        """
        Initialize the weights of the classification head.
        """
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # 1. Feature Extraction
        # Output shape: (B, C, H_feat, W_feat)
        features = self.backbone(x)

        # 2. Pooling
        # Output shape: (B, C, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Output shape: (B, C)
        flattened = pooled.flatten(1)

        # 4. Classification Head
        x = self.norm(flattened)
        x = self.drop(x)
        logits = self.fc(x)

        return logits
