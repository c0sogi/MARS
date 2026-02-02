import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config


class CoordAtt(nn.Module):
    """
    Coordinate Attention Block.
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
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class TimePreservingResNet18(nn.Module):
    """
    ResNet-18 backbone with:
    1. 1-channel input modification.
    2. Coordinate Attention injection.
    3. Asymmetric strides (2, 1) in Layer 3 and 4 to preserve temporal resolution.
    """

    def __init__(self):
        super(TimePreservingResNet18, self).__init__()
        # Load Pretrained ResNet18
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer for 1 channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize with average of pretrained weights
        with torch.no_grad():
            self.conv1.weight.data = backbone.conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Inject Coordinate Attention
        self.layer1 = self._inject_ca(backbone.layer1, 64)
        self.layer2 = self._inject_ca(backbone.layer2, 128)
        self.layer3 = self._inject_ca(backbone.layer3, 256)
        self.layer4 = self._inject_ca(backbone.layer4, 512)

        # Modify strides for Time Preservation in Layer 3 and 4
        # Original L3 stride is 2. We want (2, 1) -> Freq downsample, Time preserve
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)

        # Original L4 stride is 2. We want (2, 1)
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

    def _inject_ca(self, layer, channels):
        new_layers = []
        for block in layer:
            new_layers.append(block)
            new_layers.append(CoordAtt(channels))
        return nn.Sequential(*new_layers)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l2, l3, l4


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to aggregate temporal sequence into a single vector.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x: (Batch, Time, Feats)
        weights = self.attention(x)
        out = torch.sum(x * weights, dim=1)
        return out


class HierarchicalCRNN(nn.Module):
    """
    Hierarchical Coordinate-Attention ResNet-18 CRNN.
    Aggregates features from Layers 2, 3, and 4, projects them, and models
    temporal dynamics with a BiGRU and Attention Pooling.
    """

    def __init__(self):
        super(HierarchicalCRNN, self).__init__()

        self.backbone = TimePreservingResNet18()

        # Feature Aggregation Projection
        # L2: 128 ch, L3: 256 ch, L4: 512 ch -> Total 896
        self.projection = nn.Conv1d(896, Config.PROJECTION_DIM, kernel_size=1)

        # Temporal Modeling
        self.gru = nn.GRU(
            input_size=Config.PROJECTION_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Attention Pooling & Classifier
        self.attn_pool = AttentionPooling(Config.HIDDEN_SIZE * 2)
        self.classifier = nn.Linear(Config.HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        # x: (B, 1, F, T)
        l2, l3, l4 = self.backbone(x)

        # Hierarchical Feature Aggregation
        # Global Average Pooling over Frequency dimension (dim 2)
        # l2: (B, 128, F/8, T/8) -> (B, 128, T/8)
        f2 = torch.mean(l2, dim=2)
        f3 = torch.mean(l3, dim=2)
        f4 = torch.mean(l4, dim=2)

        # Ensure time dimensions match (should be identical due to stride settings)
        min_t = min(f2.shape[2], f3.shape[2], f4.shape[2])
        f2 = f2[:, :, :min_t]
        f3 = f3[:, :, :min_t]
        f4 = f4[:, :, :min_t]

        # Concatenate features from different depths
        combined = torch.cat([f2, f3, f4], dim=1)  # (B, 896, T)

        # Project to lower dimension
        projected = self.projection(combined)  # (B, 512, T)

        # Prepare for GRU (B, T, C)
        gru_in = projected.permute(0, 2, 1)

        # Recurrent Processing
        gru_out, _ = self.gru(gru_in)  # (B, T, 512)

        # Attention Pooling
        pooled = self.attn_pool(gru_out)  # (B, 512)

        # Classification
        logits = self.classifier(pooled)
        return logits
