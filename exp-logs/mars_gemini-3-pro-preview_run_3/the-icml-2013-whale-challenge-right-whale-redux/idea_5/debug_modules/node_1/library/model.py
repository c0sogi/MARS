import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Trainable pooling layer that generalizes Max and Average pooling.
    p=1 -> Average Pooling, p=infinity -> Max Pooling.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN gradients for p < 1
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


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design.
    Factorizes channel attention into two 1D feature encoding processes
    that aggregate features along the two spatial directions.
    """

    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # Pool spatial directions separately
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension for shared processing
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into height and width attention maps
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Compute attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention
        out = identity * a_w * a_h

        return out


class WhaleModel(nn.Module):
    """
    Right Whale Detection Model (Idea 5).
    Backbone: ConvNeXt-Small
    Attention: Coordinate Attention
    Pooling: GeM
    Head: Multi-Sample Dropout + Linear
    """

    def __init__(self, pretrained=True):
        super(WhaleModel, self).__init__()

        # 1. Backbone
        # Initialize ConvNeXt-Small without the classifier head and global pooling
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="",
        )

        # Determine feature dimension automatically
        if hasattr(self.backbone, "num_features"):
            num_features = self.backbone.num_features
        else:
            # Fallback inference of output channels
            with torch.no_grad():
                dummy = torch.randn(1, Config.IN_CHANNELS, 224, 224)
                features = self.backbone(dummy)
                num_features = features.shape[1]

        # 2. Coordinate Attention
        self.use_coord_attn = Config.USE_COORD_ATTN
        if self.use_coord_attn:
            self.coord_attn = CoordinateAttention(num_features, num_features)

        # 3. Pooling
        if Config.POOLING == "gem":
            self.global_pool = GeM()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 4. Classification Head with Multi-Sample Dropout
        self.use_ms_dropout = Config.USE_MS_DROPOUT
        self.dropout_rate = Config.DROPOUT_RATE
        self.num_classes = Config.NUM_CLASSES

        if self.use_ms_dropout:
            # Create 5 dropout layers with the same rate but different masks
            self.dropouts = nn.ModuleList(
                [nn.Dropout(self.dropout_rate) for _ in range(5)]
            )
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate)])

        self.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        # Feature Extraction
        x = self.backbone(x)  # Shape: (B, C, H, W)

        # Attention Mechanism
        if self.use_coord_attn:
            x = self.coord_attn(x)

        # Global Pooling
        x = self.global_pool(x)  # Shape: (B, C, 1, 1)
        x = x.flatten(1)  # Shape: (B, C)

        # Classification Head
        if self.use_ms_dropout:
            # Average predictions across multiple dropout masks
            logits = torch.mean(
                torch.stack([self.fc(dropout(x)) for dropout in self.dropouts]), dim=0
            )
        else:
            x = self.dropouts[0](x)
            logits = self.fc(x)

        return logits
