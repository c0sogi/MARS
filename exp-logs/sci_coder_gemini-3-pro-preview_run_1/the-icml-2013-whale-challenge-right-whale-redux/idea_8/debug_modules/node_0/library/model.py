import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from library.config import Config


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


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module.
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
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

        # C x H x 1
        x_h = self.pool_h(x)
        # C x 1 x W
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CABasicBlock(nn.Module):
    """
    ResNet BasicBlock with Coordinate Attention inserted.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # Coordinate Attention
        self.ca = CoordAtt(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Coordinate Attention
        out = self.ca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x: (Batch, Time, Features)
        weights = self.attention(x)  # (Batch, Time, 1)
        out = torch.sum(x * weights, dim=1)  # (Batch, Features)
        return out


class CoordinateAttentionCRNN(nn.Module):
    def __init__(self):
        super(CoordinateAttentionCRNN, self).__init__()

        # ------------------------------------------------------------------
        # Backbone: Time-Preserving ResNet-18 with Coordinate Attention
        # ------------------------------------------------------------------
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        # Layer 1: Stride 1
        self.layer1 = self._make_layer(CABasicBlock, 64, 2, stride=1)
        # Layer 2: Stride 2 (standard)
        self.layer2 = self._make_layer(CABasicBlock, 128, 2, stride=2)
        # Layer 3: Stride (2, 1) - Preserves Time
        self.layer3 = self._make_layer(CABasicBlock, 256, 2, stride=(2, 1))
        # Layer 4: Stride (2, 1) - Preserves Time
        self.layer4 = self._make_layer(CABasicBlock, 512, 2, stride=(2, 1))

        # ------------------------------------------------------------------
        # Temporal Modeling: BiGRU
        # ------------------------------------------------------------------
        self.gru_hidden_size = 128
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=self.gru_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # ------------------------------------------------------------------
        # Aggregation & Head
        # ------------------------------------------------------------------
        self.attn_pooling = AttentionPooling(self.gru_hidden_size * 2)
        self.fc = nn.Linear(self.gru_hidden_size * 2, Config.NUM_CLASSES)

        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        # 1. Load standard ResNet18 weights
        print("Loading ImageNet pre-trained weights for ResNet-18 backbone...")
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        state_dict = resnet18.state_dict()

        my_state_dict = self.state_dict()

        # 2. Transfer weights
        for key, param in state_dict.items():
            if key in my_state_dict:
                if my_state_dict[key].shape == param.shape:
                    my_state_dict[key].copy_(param)
                elif key == "conv1.weight":
                    # Average 3 channels to 1 channel
                    # Shape: (64, 3, 7, 7) -> (64, 1, 7, 7)
                    print("Adapting conv1 weights from 3 channels to 1 channel.")
                    my_state_dict[key].copy_(param.mean(dim=1, keepdim=True))

        # 3. Initialize new layers (CoordAtt, GRU, FC)
        # These are already initialized by default, but we can be explicit if needed.
        # PyTorch defaults are generally fine (Kaiming/Xavier).

        # Explicitly initialize the FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Output: (Batch, 512, F', T')

        # Frequency Pooling (Global Avg Pool on Freq axis)
        # We want to keep Time axis.
        x = torch.mean(x, dim=2)  # (Batch, 512, T')

        # Permute for GRU: (Batch, Time, Features)
        x = x.permute(0, 2, 1)

        # BiGRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # (Batch, Time, Hidden*2)

        # Attention Pooling
        x = self.attn_pooling(x)  # (Batch, Hidden*2)

        # Classification
        x = self.fc(x)  # (Batch, 1)

        return x
