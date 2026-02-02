import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # (N, C, H, W) -> (N, C, 1, 1)
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
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.
    Factorizes attention into two 1D feature encoding processes.
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

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Attention generation
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class WhaleClassifier(nn.Module):
    def __init__(self):
        super(WhaleClassifier, self).__init__()

        # 1. Load Backbone (EfficientNetV2-M)
        # Cite solution_lesson_node_00016
        weights = EfficientNet_V2_M_Weights.DEFAULT if Config.PRETRAINED else None
        self.backbone = efficientnet_v2_m(weights=weights)

        # 2. Modify first layer for 1-channel input
        # EfficientNetV2 features[0][0] is the first Conv2d
        if Config.IN_CHANNELS != 3:
            original_layer = self.backbone.features[0][0]
            new_layer = nn.Conv2d(
                in_channels=Config.IN_CHANNELS,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=original_layer.bias is not None,
            )

            with torch.no_grad():
                new_layer.weight[:] = original_layer.weight.sum(dim=1, keepdim=True)
                if original_layer.bias is not None:
                    new_layer.bias[:] = original_layer.bias

            self.backbone.features[0][0] = new_layer

        # Feature dimension for EfficientNetV2-M is 1280
        self.num_features = self.backbone.classifier[1].in_features

        # 3. Pooling
        self.use_gem = Config.USE_GEM_POOL
        if self.use_gem:
            self.avgpool = GeM()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 4. Classifier Head
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features (N, 1280, H, W)
        x = self.backbone.features(x)

        # Pooling (N, 1280, 1, 1)
        x = self.avgpool(x)

        # Flatten (N, 1280)
        x = torch.flatten(x, 1)

        # Classification (N, 1)
        x = self.fc(x)

        return x
