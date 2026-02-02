import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from library.layers import CoordinateAttention, SEBlock, AttentionPooling


class CABasicBlock(nn.Module):
    """
    Residual Block equipped with Coordinate Attention.
    Replaces the standard ResNet BasicBlock.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(CABasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("CABasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in CABasicBlock")

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)

        # Coordinate Attention inserted here
        self.ca = CoordinateAttention(planes, reduction=32)

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


class AdaptiveResNetCRNN(nn.Module):
    """
    Ensemble of High-Resolution Spectrally-Adaptive Hierarchical CA-ResNet-18 CRNNs.

    Architecture:
    1. Backbone: ResNet-18 with Coordinate Attention Blocks.
       - Stem modified for 1-channel input and time-preservation (Stride 2,1).
       - Layers 3 and 4 use asymmetric strides (2,1) to preserve time resolution.
    2. Hierarchical Feature Extraction:
       - Extracts features from Layers 2, 3, and 4.
    3. Adaptive Spectral Pooling:
       - Layer 2 -> 4 Freq Bins
       - Layer 3 -> 2 Freq Bins
       - Layer 4 -> 1 Freq Bin
    4. Fusion:
       - Concatenation -> 1x1 Conv Bottleneck -> SEBlock.
    5. Head:
       - Bi-Directional GRU -> Attention Pooling -> Classifier.
    """

    def __init__(self):
        super(AdaptiveResNetCRNN, self).__init__()

        # 1. Load Pretrained ResNet-18
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 2. Modify Stem
        # Original: Conv2d(3, 64, 7, stride=2, padding=3)
        # New: Conv2d(1, 64, 7, stride=(2, 1), padding=3) to preserve time
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=(2, 1), padding=3, bias=False
        )

        # Initialize 1-channel weights by averaging the 3-channel pretrained weights
        with torch.no_grad():
            self.conv1.weight.data = (
                backbone.conv1.weight.data.sum(dim=1, keepdim=True) / 3.0
            )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        # Remove MaxPool to preserve temporal resolution for short clips
        self.maxpool = nn.Identity()

        # 3. Build Layers with CABasicBlock and Custom Strides
        self.layer1 = self._make_layer_from_backbone(
            backbone.layer1, stride_override=None
        )
        self.layer2 = self._make_layer_from_backbone(
            backbone.layer2, stride_override=None
        )  # Standard stride 2
        self.layer3 = self._make_layer_from_backbone(
            backbone.layer3, stride_override=(2, 1)
        )  # Preserves time
        self.layer4 = self._make_layer_from_backbone(
            backbone.layer4, stride_override=(2, 1)
        )  # Preserves time

        # 4. Fusion & Bottleneck
        # Calculate fused dimension
        # L2: 128 ch * 4 bins = 512
        # L3: 256 ch * 2 bins = 512
        # L4: 512 ch * 1 bin  = 512
        # Total = 1536
        fused_dim = 1536
        bottleneck_dim = 256

        self.bottleneck = nn.Sequential(
            nn.Conv2d(fused_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            SEBlock(bottleneck_dim, reduction=16),
        )

        # 5. Recurrent Head
        self.gru = nn.GRU(
            input_size=bottleneck_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_pooling = AttentionPooling(input_dim=256)  # 128 * 2
        self.fc = nn.Linear(256, 1)

    def _make_layer_from_backbone(self, original_layer, stride_override=None):
        """
        Reconstructs a ResNet layer using CABasicBlock, copying weights,
        and optionally overriding the stride of the first block.
        """
        layers = []
        for i, block in enumerate(original_layer):
            # Determine stride
            stride = block.stride
            if i == 0 and stride_override is not None:
                stride = stride_override

            # Handle Downsample layer
            downsample = None
            if block.downsample is not None or (i == 0 and stride_override is not None):
                # Reconstruct downsample with new stride if needed
                # Standard ResNet downsample is Conv 1x1 -> BN
                in_ch = block.conv1.in_channels
                out_ch = block.conv1.out_channels * block.expansion

                # If modifying stride, we must ensure downsample matches
                ds_stride = stride

                # Copy weights if shapes match (stride doesn't affect weight shape)
                ds_conv = nn.Conv2d(
                    in_ch, out_ch, kernel_size=1, stride=ds_stride, bias=False
                )
                ds_bn = nn.BatchNorm2d(out_ch)

                if block.downsample is not None:
                    ds_conv.weight.data = block.downsample[0].weight.data
                    ds_bn.weight.data = block.downsample[1].weight.data
                    ds_bn.bias.data = block.downsample[1].bias.data
                    ds_bn.running_mean.data = block.downsample[1].running_mean.data
                    ds_bn.running_var.data = block.downsample[1].running_var.data

                downsample = nn.Sequential(ds_conv, ds_bn)

            # Create CABasicBlock
            new_block = CABasicBlock(
                block.conv1.in_channels,
                block.conv1.out_channels,
                stride=stride,
                downsample=downsample,
            )

            # Copy weights
            new_block.conv1.weight.data = block.conv1.weight.data
            new_block.bn1.weight.data = block.bn1.weight.data
            new_block.bn1.bias.data = block.bn1.bias.data
            new_block.bn1.running_mean.data = block.bn1.running_mean.data
            new_block.bn1.running_var.data = block.bn1.running_var.data

            new_block.conv2.weight.data = block.conv2.weight.data
            new_block.bn2.weight.data = block.bn2.weight.data
            new_block.bn2.bias.data = block.bn2.bias.data
            new_block.bn2.running_mean.data = block.bn2.running_mean.data
            new_block.bn2.running_var.data = block.bn2.running_var.data

            layers.append(new_block)

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, F, T)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Hierarchical Features
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # Adaptive Spectral Pooling & Flattening
        # L2: (B, 128, F2, T) -> Pool F to 4 -> (B, 128, 4, T) -> (B, 512, 1, T)
        f2 = torch.nn.functional.adaptive_avg_pool2d(l2, (4, None))
        b, c, f, t = f2.size()
        f2 = f2.view(b, c * f, 1, t)

        # L3: (B, 256, F3, T) -> Pool F to 2 -> (B, 256, 2, T) -> (B, 512, 1, T)
        f3 = torch.nn.functional.adaptive_avg_pool2d(l3, (2, None))
        b, c, f, t = f3.size()
        f3 = f3.view(b, c * f, 1, t)

        # L4: (B, 512, F4, T) -> Pool F to 1 -> (B, 512, 1, T) -> (B, 512, 1, T)
        f4 = torch.nn.functional.adaptive_avg_pool2d(l4, (1, None))
        b, c, f, t = f4.size()
        f4 = f4.view(b, c * f, 1, t)

        # Concatenate along channel dimension
        # Result: (B, 1536, 1, T)
        # Note: Time dimensions must match.
        # With stride settings:
        # Input (T) -> Conv1(s2,1) -> T.
        # L1(s1) -> T.
        # L2(s2) -> T/2.
        # L3(s2,1) -> T/2.
        # L4(s2,1) -> T/2.
        # So L2, L3, L4 should have same T.
        x_fused = torch.cat([f2, f3, f4], dim=1)

        # Bottleneck & SE
        # (B, 1536, 1, T) -> (B, 256, 1, T)
        x_fused = self.bottleneck(x_fused)

        # Prepare for RNN
        # (B, C, 1, T) -> (B, C, T) -> (B, T, C)
        x_rnn = x_fused.squeeze(2).permute(0, 2, 1)

        # Bi-GRU
        # (B, T, 256)
        rnn_out, _ = self.gru(x_rnn)

        # Attention Pooling
        # (B, 256)
        pool_out = self.attn_pooling(rnn_out)

        # Classifier
        logits = self.fc(pool_out)

        return logits
