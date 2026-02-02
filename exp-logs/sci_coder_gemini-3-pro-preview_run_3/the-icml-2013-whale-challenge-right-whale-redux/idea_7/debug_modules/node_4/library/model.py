import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import ModelConfig


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the input feature map.
    Formula: f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp to avoid numerical instability with pow
        # Average pool over spatial dimensions (H, W)
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


class CoordAtt(nn.Module):
    """
    Coordinate Attention Block.
    Factorizes attention into two 1D feature encoding processes to preserve
    positional information.
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # Pool vertically and horizontally
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, W) -> (B, C, W, 1)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into H and W
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W)

        # Compute attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention
        out = identity * a_h * a_w

        return out


class WhaleDetector(nn.Module):
    """
    Main model class for Right Whale Call Detection.
    Integrates ConvNeXt backbone, Coordinate Attention, and GeM Pooling.
    """

    def __init__(self):
        super(WhaleDetector, self).__init__()

        # Initialize Backbone (ConvNeXt-Small)
        # global_pool='' ensures we get spatial feature maps (B, C, H, W)
        # num_classes=0 removes the default linear head
        self.backbone = timm.create_model(
            ModelConfig.model_name,
            pretrained=ModelConfig.pretrained,
            in_chans=ModelConfig.in_chans,
            num_classes=0,
            global_pool="",
            drop_path_rate=ModelConfig.drop_path_rate,
        )

        self.num_features = self.backbone.num_features

        # Coordinate Attention
        if ModelConfig.use_coord_att:
            self.coord_att = CoordAtt(self.num_features)
        else:
            self.coord_att = nn.Identity()

        # Generalized Mean Pooling
        if ModelConfig.use_gem:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.drop = nn.Dropout(ModelConfig.drop_rate)
        self.fc = nn.Linear(self.num_features, ModelConfig.num_classes)

    def forward(self, x):
        # 1. Backbone Feature Extraction
        # Input: (B, 1, F, T) -> Output: (B, C, H, W)
        x = self.backbone(x)

        # 2. Coordinate Attention
        # Refines features using positional information
        x = self.coord_att(x)

        # 3. Pooling
        # (B, C, H, W) -> (B, C, 1, 1)
        x = self.pooling(x)
        x = x.flatten(1)  # (B, C)

        # 4. Classification Head
        x = self.drop(x)
        x = self.fc(x)

        return x
