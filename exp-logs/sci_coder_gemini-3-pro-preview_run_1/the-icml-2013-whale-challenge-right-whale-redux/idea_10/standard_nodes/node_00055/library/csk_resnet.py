import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CoordinateAttentionFusion(nn.Module):
    """
    Computes attention weights for fusing multiple branches using
    Coordinate Attention logic (pooling along H and W separately).
    """

    def __init__(self, channels, reduction=16, num_branches=2):
        super(CoordinateAttentionFusion, self).__init__()
        self.num_branches = num_branches

        # Reduction dimension
        mid_channels = max(8, channels // reduction)

        self.conv_reduce = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # Expansion to num_branches * channels
        self.conv_h = nn.Conv2d(
            mid_channels, channels * num_branches, kernel_size=1, bias=False
        )
        self.conv_w = nn.Conv2d(
            mid_channels, channels * num_branches, kernel_size=1, bias=False
        )

    def forward(self, x):
        """
        x: Input tensor of shape (N, C, H, W) representing the sum of branches.
        Returns: Attention weights of shape (N, num_branches, C, H, W)
        """
        n, c, h, w = x.size()

        # 1. Coordinate Pooling
        # Pool H -> (N, C, H, 1)
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        # Pool W -> (N, C, 1, W)
        x_w = F.adaptive_avg_pool2d(x, (1, w))

        # 2. Concat for shared processing
        # Permute x_w to (N, C, W, 1) to stack with x_h along spatial dim
        x_w_perm = x_w.permute(0, 1, 3, 2)

        # Concat: (N, C, H+W, 1)
        y = torch.cat([x_h, x_w_perm], dim=2)

        # 3. Reduction
        y = self.conv_reduce(y)
        y = self.bn(y)
        y = self.act(y)

        # 4. Split
        x_h_feat, x_w_feat = torch.split(y, [h, w], dim=2)

        # Restore x_w_feat shape: (N, mid, W, 1) -> (N, mid, 1, W)
        x_w_feat = x_w_feat.permute(0, 1, 3, 2)

        # 5. Expansion to generate weights per branch
        # (N, K*C, H, 1)
        attn_h = self.conv_h(x_h_feat)
        # (N, K*C, 1, W)
        attn_w = self.conv_w(x_w_feat)

        # Reshape to (N, K, C, H, 1) and (N, K, C, 1, W)
        attn_h = attn_h.view(n, self.num_branches, c, h, 1)
        attn_w = attn_w.view(n, self.num_branches, c, 1, w)

        # 6. Combine and Softmax
        # We sum the attention scores from H and W context
        # Then apply softmax across the 'branches' dimension (dim 1)
        attn = attn_h + attn_w
        attn = F.softmax(attn, dim=1)

        return attn


class CSKConv(nn.Module):
    """
    Coordinate-Selective-Kernel Convolution.
    Features two branches with different kernel sizes/dilations, fused
    via Coordinate Attention weights.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=1):
        super(CSKConv, self).__init__()

        # Branch 1: Standard 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: Dilated 3x3 (Effective 5x5 context)
        # We increase dilation to capture larger context
        eff_dilation = dilation * 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=eff_dilation,
                dilation=eff_dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = CoordinateAttentionFusion(
            out_channels, reduction=16, num_branches=2
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        # Stack branches: (N, 2, C, H, W)
        branches = torch.stack([b1, b2], dim=1)

        # Sum for attention calculation
        u = b1 + b2

        # Get weights: (N, 2, C, H, W)
        weights = self.fusion(u)

        # Weighted sum
        out = torch.sum(branches * weights, dim=1)
        return out


class CSKBlock(nn.Module):
    """
    ResNet BasicBlock where the second convolution is replaced by CSKConv.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CSKBlock, self).__init__()

        # 1st Conv: Standard 3x3
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 2nd Conv: CSK Conv (Multi-scale fusion)
        # Note: stride is handled in conv1, so stride here is 1
        self.conv2 = CSKConv(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

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
        w = self.attention(x)  # (Batch, Time, 1)
        out = torch.sum(x * w, dim=1)  # (Batch, Features)
        return out


class CSKResNet18CRNN(nn.Module):
    def __init__(self):
        super(CSKResNet18CRNN, self).__init__()

        # ==========================
        # 1. Backbone: CSK-ResNet18
        # ==========================
        self.inplanes = 64

        # Stem
        # Adapt first conv for 1-channel input if needed, but standard ResNet is 3.
        # We define it explicitly for IN_CHANNELS.
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        # Layer 1: 64 filters, stride 1
        self.layer1 = self._make_layer(CSKBlock, 64, 2, stride=1)
        # Layer 2: 128 filters, stride 2
        self.layer2 = self._make_layer(CSKBlock, 128, 2, stride=2)
        # Layer 3: 256 filters, stride (2, 1) -> Preserves Time
        self.layer3 = self._make_layer(CSKBlock, 256, 2, stride=(2, 1))
        # Layer 4: 512 filters, stride (2, 1) -> Preserves Time
        self.layer4 = self._make_layer(CSKBlock, 512, 2, stride=(2, 1))

        # ==========================
        # 2. Temporal Modeling
        # ==========================
        # Bi-GRU
        # Input size is 512 (channels from layer4)
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # ==========================
        # 3. Aggregation & Head
        # ==========================
        # Attention Pooling (512 features from Bi-GRU)
        self.attn_pool = AttentionPooling(512)

        # Classifier
        self.fc = nn.Linear(512, Config.NUM_CLASSES)

        # Weight Initialization
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # Check stride type (int or tuple)
        stride_h = stride[0] if isinstance(stride, tuple) else stride
        stride_w = stride[1] if isinstance(stride, tuple) else stride

        if stride_h != 1 or stride_w != 1 or self.inplanes != planes * block.expansion:
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (N, 1, F, T)

        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Shape: (N, 512, F', T')

        # Frequency Pooling
        # Pool completely along Frequency axis to get (N, 512, 1, T')
        x = torch.mean(x, dim=2, keepdim=True)

        # Prepare for RNN
        # (N, 512, 1, T') -> (N, 512, T')
        x = x.squeeze(2)
        # (N, 512, T') -> (N, T', 512)
        x = x.permute(0, 2, 1)

        # RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (N, T', 512)

        # Attention Pooling
        x = self.attn_pool(x)  # (N, 512)

        # Classifier
        x = self.fc(x)  # (N, 1)

        return x
