import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """
    Wraps a torchvision BasicBlock to add Squeeze-and-Excitation.
    Replicates the logic of BasicBlock but inserts SE before the residual addition.
    """

    def __init__(self, original_block, reduction=16):
        super(SEBasicBlock, self).__init__()
        # Adopt layers from the original pretrained block
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride

        # The number of output channels is defined in bn2.num_features
        self.se = SEBlock(original_block.bn2.num_features, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Squeeze-and-Excitation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SeismicHybridModel(nn.Module):
    """
    Idea 8: Stabilized SE-ResNet34 Hybrid with Dual-Domain Feature Injection.
    Combines a channel-adapted SE-ResNet34 for spectrograms with a Wide MLP for tabular features.
    """

    def __init__(self, num_tabular_features):
        super(SeismicHybridModel, self).__init__()

        # ------------------------------------------------------------------
        # Branch 1: SE-ResNet34 Spectrogram Encoder
        # ------------------------------------------------------------------
        # Load Pretrained ResNet34
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.backbone = resnet34(weights=weights)

        # 1. Adapt First Convolution Layer for 10 Channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 10
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize weights: Average RGB (dim 1) and replicate 10 times
        # This preserves the spatial filters learned from ImageNet while adapting to 10 sensors
        with torch.no_grad():
            # shape: [64, 3, 7, 7]
            w = original_conv1.weight
            # Average over input channels -> [64, 1, 7, 7]
            w_avg = torch.mean(w, dim=1, keepdim=True)
            # Replicate -> [64, 10, 7, 7]
            w_new = w_avg.repeat(1, Config.IN_CHANNELS, 1, 1)
            new_conv1.weight.copy_(w_new)

        self.backbone.conv1 = new_conv1

        # 2. Inject SE Blocks into Layers
        # Iterate through the ResNet stages and wrap each BasicBlock
        if Config.USE_SE:
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                layer_container = getattr(self.backbone, layer_name)
                for i, block in enumerate(layer_container):
                    # Replace BasicBlock with SEBasicBlock
                    # We pass the existing block to preserve pretrained weights
                    se_block = SEBasicBlock(block, reduction=16)
                    layer_container[i] = se_block

        # Remove original FC and AvgPool (we use custom GAP and Head)
        del self.backbone.fc
        del self.backbone.avgpool

        # Calculate CNN output dimension
        # ResNet34 layer4 output channels is 512
        self.cnn_out_dim = 512

        # ------------------------------------------------------------------
        # Branch 2: Dual-Domain Wide MLP
        # ------------------------------------------------------------------
        layers = []
        in_dim = num_tabular_features

        for hidden_dim in Config.MLP_HIDDEN_LAYERS:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.MLP_DROPOUT))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.mlp_out_dim = Config.MLP_HIDDEN_LAYERS[-1]

        # ------------------------------------------------------------------
        # Fusion Head
        # ------------------------------------------------------------------
        fusion_dim = self.cnn_out_dim + self.mlp_out_dim
        self.regressor = nn.Linear(fusion_dim, 1)

    def forward(self, spectrogram, tabular):
        """
        Args:
            spectrogram (torch.Tensor): Shape [Batch, 10, 128, Time]
            tabular (torch.Tensor): Shape [Batch, Num_Features]
        """
        # --- CNN Branch ---
        x = self.backbone.conv1(spectrogram)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global Average Pooling
        # x shape: [Batch, 512, H, W] -> [Batch, 512]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # --- MLP Branch ---
        # tab shape: [Batch, Num_Features] -> [Batch, 128]
        tab = self.mlp(tabular)

        # --- Fusion ---
        combined = torch.cat((x, tab), dim=1)
        out = self.regressor(combined)

        return out
