import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.
    formula: f(x) = (1/|x| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Apply average pooling on x^p, then take the (1/p)-th root
        # x shape: (B, C, H, W)
        # Output shape: (B, C, 1, 1)
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


class h_sigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.
    Factorizes attention into two 1D encoding processes (H and W) to preserve
    positional information.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        identity = x
        n, c, h, w = x.size()

        # Pool along height -> (B, C, H, 1)
        x_h = self.pool_h(x)
        # Pool along width -> (B, C, 1, W) -> Permute to (B, C, W, 1) for concatenation
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)

        # Shared 1x1 Conv -> BN -> Non-linearity
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into H and W components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)

        # Generate attention maps
        a_h = self.sigmoid(self.conv_h(x_h))  # (B, C, H, 1)
        a_w = self.sigmoid(self.conv_w(x_w))  # (B, C, 1, W)

        # Apply attention
        out = identity * a_h * a_w

        return out


class WhaleDetector(nn.Module):
    """
    Right Whale Detection Model.
    Backbone: ConvNeXt-Tiny
    Attention: Coordinate Attention
    Pooling: Generalized Mean (GeM)
    Head: Linear Classifier
    """

    def __init__(self, pretrained=True):
        super(WhaleDetector, self).__init__()

        # 1. Backbone: ConvNeXt Tiny
        # in_chans=1 allows timm to adapt the first layer weights (summing RGB weights)
        # num_classes=0 and global_pool='' gives us the raw feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="",
        )

        # Get the number of output channels from the backbone
        # For ConvNeXt-Tiny, this is typically 768
        self.num_features = self.backbone.num_features

        # 2. Coordinate Attention
        # Applied to the feature maps extracted by the backbone
        if Config.USE_COORDINATE_ATTENTION:
            self.attn = CoordinateAttention(self.num_features, reduction=32)
        else:
            self.attn = nn.Identity()

        # 3. Pooling
        if Config.USE_GEM_POOL:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classifier Head
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        # x shape: (B, 1, H, W)

        # Extract features
        features = self.backbone(x)  # (B, C, H', W')

        # Apply Attention
        features = self.attn(features)

        # Pooling
        # GeM output is (B, C, 1, 1), flatten to (B, C)
        embedding = self.pool(features).flatten(1)

        # Classification
        logits = self.fc(embedding)  # (B, 1)

        return logits
