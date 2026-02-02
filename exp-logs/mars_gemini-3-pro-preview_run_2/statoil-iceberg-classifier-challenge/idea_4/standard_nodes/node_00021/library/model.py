import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseLayer(nn.Module):
    """
    A single layer within a Dense Block.
    Structure: BN -> ReLU -> 3x3 Conv
    Output: Concatenation of Input and Conv Output
    """

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        # Bottleneck structure could be added here, but standard DenseNet-BC
        # uses BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3).
        # For this small image size (75x75), we stick to the simpler Basic Dense Layer
        # to preserve spatial details: BN-ReLU-Conv(3x3).
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    """
    A block consisting of multiple DenseLayers.
    """

    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """
    Transition layer between Dense Blocks to reduce dimensions.
    Structure: BN -> ReLU -> 1x1 Conv -> 2x2 AvgPool
    """

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DCHN(nn.Module):
    """
    Densely Connected Hybrid Network (DCHN)

    Combines a custom DenseNet visual backbone with a metadata processing branch.
    """

    def __init__(self):
        super(DCHN, self).__init__()

        # --- Configuration ---
        growth_rate = Config.GROWTH_RATE
        block_config = Config.BLOCK_CONFIG
        compression = Config.COMPRESSION

        # Initial number of features after the stem
        # Usually 2 * growth_rate is a good starting point
        num_init_features = 2 * growth_rate

        # --- Visual Branch (Custom DenseNet) ---

        # Stem: 3x3 Conv, stride 1, padding 1 to preserve 75x75 resolution
        self.stem = nn.Conv2d(
            Config.IN_CHANNELS, num_init_features, kernel_size=3, padding=1, bias=False
        )

        # Dense Blocks and Transitions
        self.features = nn.Sequential()
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers, in_channels=num_features, growth_rate=growth_rate
            )
            self.features.add_module(f"denseblock{i+1}", block)

            # Update num_features: input + layers * growth_rate
            num_features = num_features + num_layers * growth_rate

            # Add Transition Layer after each block EXCEPT the last one
            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = TransitionLayer(
                    in_channels=num_features, out_channels=out_features
                )
                self.features.add_module(f"transition{i+1}", trans)
                num_features = out_features

        # Final Batch Norm and Global Average Pooling
        self.final_bn = nn.BatchNorm2d(num_features)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.visual_out_dim = num_features

        # --- Metadata Branch ---
        self.meta_hidden_dim = Config.META_HIDDEN_DIM
        self.meta_branch = nn.Sequential(
            nn.Linear(1, self.meta_hidden_dim),
            nn.BatchNorm1d(self.meta_hidden_dim),
            nn.ReLU(),
        )

        # --- Fusion Head ---
        # Concatenated dimension
        fusion_input_dim = self.visual_out_dim + self.meta_hidden_dim

        # Classification Head: Dense -> BN -> ReLU -> Dropout -> Dense -> Sigmoid
        # We choose an intermediate dense size of 256
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, Config.NUM_CLASSES),
            nn.Sigmoid(),
        )

    def forward(self, x_img, x_angle):
        """
        Args:
            x_img: (Batch, 3, 75, 75)
            x_angle: (Batch,) or (Batch, 1)
        """
        # 1. Visual Branch
        x = self.stem(x_img)
        x = self.features(x)
        x = F.relu(self.final_bn(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)  # (Batch, visual_out_dim)

        # 2. Metadata Branch
        # Ensure angle has shape (Batch, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)

        m = self.meta_branch(x_angle)  # (Batch, meta_hidden_dim)

        # 3. Fusion
        combined = torch.cat([x, m], dim=1)

        # 4. Classification
        out = self.fusion_head(combined)

        return out
