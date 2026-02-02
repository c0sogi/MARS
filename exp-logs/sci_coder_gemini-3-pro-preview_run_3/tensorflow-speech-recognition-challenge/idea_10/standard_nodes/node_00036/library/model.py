import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    CNN1D_FILTERS,
    CNN1D_KERNELS,
    CNN1D_STRIDES,
    RESNET_LAYERS,
    RESNET_STRIDES,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    RNN_DROPOUT,
    BIDIRECTIONAL,
    ATTN_NUM_HEADS,
    ATTN_HIDDEN_DIM,
    NUM_CLASSES,
)


class SKConv(nn.Module):
    """
    Selective Kernel Convolution.
    Aggregates information from multiple kernel branches using channel-wise attention.
    """

    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """
        Args:
            features (int): Number of input/output channels.
            M (int): Number of branches.
            G (int): Group size for grouped convolutions.
            r (int): Reduction ratio for the attention fc layer.
            stride (int/tuple): Stride for the convolutions.
            L (int): Minimum dimension for the attention fc layer.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])

        for i in range(M):
            # Branch i uses a different dilation to simulate different kernel sizes
            # Branch 0: dilation 1 (3x3)
            # Branch 1: dilation 2 (approx 5x5)
            dilation = 1 + i
            padding = 1 + i
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=G,
                        bias=False,
                    ),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                )
            )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(features, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features, bias=False))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        # 1. Split: Compute output of each branch
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)  # (B, M, C, H, W)

        # 2. Fuse: Sum over branches
        U = torch.sum(feats, dim=1)  # (B, C, H, W)

        # 3. Select: Channel-wise attention
        S = self.gap(U).view(batch_size, -1)  # (B, C)
        Z = self.fc(S)  # (B, d)

        weights = [fc(Z) for fc in self.fcs]
        weights = torch.stack(weights, dim=1)  # (B, M, C)
        weights = self.softmax(weights)  # (B, M, C)

        # Reshape for broadcasting: (B, M, C, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # Apply weights to branches and sum
        V = torch.sum(feats * weights, dim=1)  # (B, C, H, W)

        return V


class SKBasicBlock(nn.Module):
    """
    ResNet BasicBlock modification using SKConv.
    The first conv handles stride/channel projection.
    The second conv is replaced by SKConv for adaptive context.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SKBasicBlock, self).__init__()
        # Conv1: Standard 3x3 to handle stride and channel change
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Conv2: SKConv (stride 1, keeping dimensions)
        self.conv2 = SKConv(planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Stream2D_SKResNet(nn.Module):
    """
    Stream 1: SK-ResNet34 Backbone for 2D Spectrograms.
    """

    def __init__(self, layers=RESNET_LAYERS, strides=RESNET_STRIDES):
        super(Stream2D_SKResNet, self).__init__()
        self.inplanes = 64

        # Audio-friendly Stem: Stride 2 to reduce F/T slightly but preserve more than ImageNet stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self._make_layer(SKBasicBlock, 64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(SKBasicBlock, 128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(SKBasicBlock, 256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(SKBasicBlock, 512, layers[3], stride=strides[3])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # Handle tuple stride for downsample layer
        is_strided = (
            (stride != 1)
            if isinstance(stride, int)
            else (stride[0] != 1 or stride[1] != 1)
        )

        if is_strided or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 3, F, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output: (B, 512, F', T')
        return x


class Stream1D_CNN(nn.Module):
    """
    Stream 2: Deep 1D CNN for Raw Waveform.
    """

    def __init__(
        self, filters=CNN1D_FILTERS, kernels=CNN1D_KERNELS, strides=CNN1D_STRIDES
    ):
        super(Stream1D_CNN, self).__init__()
        layers = []
        in_channels = 1

        for out_channels, k, s in zip(filters, kernels, strides):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=s,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, T_raw)
        return self.net(x)


class MultiHeadAttentionPooling(nn.Module):
    """
    Aggregates temporal features using multiple attention heads.
    """

    def __init__(self, input_dim, num_heads=ATTN_NUM_HEADS, hidden_dim=ATTN_HIDDEN_DIM):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Attention scoring mechanism: V -> tanh -> W -> scores
        self.attn_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, x):
        # x: (B, T, C)

        # Compute attention scores
        # scores: (B, T, num_heads)
        scores = self.attn_fc(x)

        # Normalize over time dimension
        weights = F.softmax(scores, dim=1)  # (B, T, num_heads)

        # Weighted sum for each head
        # x: (B, T, C) -> (B, T, 1, C)
        # weights: (B, T, num_heads) -> (B, T, num_heads, 1)
        x_expanded = x.unsqueeze(2)
        weights_expanded = weights.unsqueeze(-1)

        # context: (B, num_heads, C)
        context = torch.sum(x_expanded * weights_expanded, dim=1)

        # Flatten heads: (B, num_heads * C)
        output = context.view(context.size(0), -1)

        return output


class HybridDualStreamCRNN(nn.Module):
    """
    Hybrid 1D-2D Dual-Stream CRNN.
    Fuses SK-ResNet34 (Spectral) and 1D CNN (Temporal) features.
    """

    def __init__(self):
        super(HybridDualStreamCRNN, self).__init__()

        # --- Stream 1: 2D Spectral ---
        self.stream2d = Stream2D_SKResNet()

        # Calculate Stream 1 output dimension
        # ResNet34 ends with 512 channels.
        # Assuming standard input height 64 and standard downsampling:
        # Stem: /4 -> 16
        # L1: /1 -> 16
        # L2: /2 -> 8
        # L3: /2 -> 4
        # L4: /1 -> 4
        # Flattened dim = 512 * 4 = 2048
        self.dim_2d = 512 * 4

        # --- Stream 2: 1D Temporal ---
        self.stream1d = Stream1D_CNN()
        self.dim_1d = CNN1D_FILTERS[-1]  # 256

        # --- Fusion & RNN ---
        fusion_dim = self.dim_2d + self.dim_1d

        self.rnn = nn.GRU(
            input_size=fusion_dim,
            hidden_size=RNN_HIDDEN_DIM,
            num_layers=RNN_NUM_LAYERS,
            dropout=RNN_DROPOUT if RNN_NUM_LAYERS > 1 else 0,
            bidirectional=BIDIRECTIONAL,
            batch_first=True,
        )

        rnn_out_dim = RNN_HIDDEN_DIM * 2 if BIDIRECTIONAL else RNN_HIDDEN_DIM

        # --- Head ---
        self.attn_pooling = MultiHeadAttentionPooling(rnn_out_dim)

        # Classifier input is (num_heads * rnn_out_dim)
        clf_in_dim = ATTN_NUM_HEADS * rnn_out_dim
        self.classifier = nn.Linear(clf_in_dim, NUM_CLASSES)

    def forward(self, spec, wave):
        # spec: (B, 3, F, T)
        # wave: (B, 1, T_raw)

        # 1. Process Streams
        feat_2d = self.stream2d(spec)  # (B, 512, F', T_2d)
        feat_1d = self.stream1d(wave)  # (B, 256, T_1d)

        # 2. Prepare 2D features
        # Flatten frequency dimension
        B, C2, F2, T2 = feat_2d.shape
        feat_2d = feat_2d.view(B, C2 * F2, T2)  # (B, 2048, T_2d)

        # 3. Align Temporal Dimensions
        # We upsample the 2D features to match the 1D features' time dimension
        # feat_1d is usually T=100. feat_2d might be T=6 or T=25 depending on strides.
        target_time = feat_1d.size(2)

        if feat_2d.size(2) != target_time:
            feat_2d = F.interpolate(
                feat_2d, size=target_time, mode="linear", align_corners=False
            )

        # 4. Concatenate
        # (B, 2048, T) + (B, 256, T) -> (B, 2304, T)
        fused = torch.cat([feat_2d, feat_1d], dim=1)

        # Permute for RNN: (B, T, Features)
        fused = fused.permute(0, 2, 1)

        # 5. Sequence Modeling
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(fused)  # (B, T, H*2)

        # 6. Attention Pooling
        pooled = self.attn_pooling(rnn_out)  # (B, Heads * H*2)

        # 7. Classification
        logits = self.classifier(pooled)

        return logits
