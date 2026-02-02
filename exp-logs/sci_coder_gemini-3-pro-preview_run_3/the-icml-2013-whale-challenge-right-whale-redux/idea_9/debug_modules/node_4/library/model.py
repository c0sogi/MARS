import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import ModelConfig


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a pooling parameter 'p' to interpolate between Average Pooling (p=1)
    and Max Pooling (p=infinity). Effective for weak supervision.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN with fractional powers on negative values
        # This implicitly acts like a ReLU for negative activations before pooling
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.gem.__repr__()
            + "("
            + self.p.__repr__()
            + ", eps="
            + str(self.eps)
            + ")"
        )


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.
    Factorizes attention into two 1D encoding processes to capture
    long-range dependencies along one spatial direction while preserving
    precise positional information along the other.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
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

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)

        # Shared Conv for reduction
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Attention generation
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class WhaleConvNeXt(nn.Module):
    """
    Whale Call Detection Model.
    Backbone: ConvNeXt-Small (Pretrained)
    Attention: Coordinate Attention
    Pooling: GeM
    """

    def __init__(self):
        super(WhaleConvNeXt, self).__init__()

        # Create Backbone
        # num_classes=0 and global_pool='' returns the feature map (B, C, H, W)
        # in_chans=1 allows timm to adapt the first layer weights (summing RGB)
        self.backbone = timm.create_model(
            ModelConfig.model_name,
            pretrained=ModelConfig.pretrained,
            in_chans=ModelConfig.in_chans,
            num_classes=0,
            global_pool="",
        )

        # Determine the number of output channels from the backbone
        # We run a dummy forward pass to be robust against model variations
        with torch.no_grad():
            dummy_input = torch.randn(1, ModelConfig.in_chans, 224, 224)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # Coordinate Attention
        if ModelConfig.use_coordinate_attention:
            self.coord_att = CoordinateAttention(self.num_features)
        else:
            self.coord_att = nn.Identity()

        # Pooling Layer
        if ModelConfig.pooling_type == "gem":
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.drop = nn.Dropout(ModelConfig.drop_rate)
        self.fc = nn.Linear(self.num_features, ModelConfig.num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 1, F, T)
        """
        # Extract features (B, C, H', W')
        x = self.backbone(x)

        # Apply Attention
        x = self.coord_att(x)

        # Pooling -> (B, C, 1, 1)
        x = self.pool(x)

        # Flatten -> (B, C)
        x = x.flatten(1)

        # Head
        x = self.drop(x)
        x = self.fc(x)

        return x


def get_model():
    return WhaleConvNeXt()
