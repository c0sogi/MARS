import torch
import torch.nn as nn
from library.config import Config


class SpatialDropoutResBlock(nn.Module):
    """
    A Residual Block with Spatial Dropout (Dropout2d) inserted between convolutions.
    This acts as a structural regularizer for the CNN feature maps.
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(SpatialDropoutResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Spatial Dropout drops entire channels, promoting independence
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SRNResNet(nn.Module):
    """
    Spatially-Regularized Narrow ResNet (SRN-ResNet).
    Designed for small-scale radar data with high noise.
    Features:
    - Narrow channel width (capped at 128)
    - Spatial Dropout regularization
    - Global Max Pooling
    - Late Fusion with Incidence Angle
    """

    def __init__(self):
        super(SRNResNet, self).__init__()

        # Load hyperparameters from Config
        stem_filters = Config.STEM_FILTERS
        stage_filters = Config.STAGE_FILTERS
        spatial_dropout = Config.SPATIAL_DROPOUT_RATE
        classifier_dropout = Config.CLASSIFIER_DROPOUT_RATE
        hidden_dim = Config.CLASSIFIER_HIDDEN_DIM
        input_channels = Config.IN_CHANNELS

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, stem_filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_filters),
            nn.ReLU(inplace=True),
        )

        # 2. Residual Backbone
        # Stage 1: 64 -> 64 (Stride 1)
        self.layer1 = self._make_layer(
            stem_filters, stage_filters[0], stride=1, dropout_rate=spatial_dropout
        )

        # Stage 2: 64 -> 64 (Stride 2 for downsampling)
        self.layer2 = self._make_layer(
            stage_filters[0], stage_filters[1], stride=2, dropout_rate=spatial_dropout
        )

        # Stage 3: 64 -> 128 (Stride 2 for downsampling)
        self.layer3 = self._make_layer(
            stage_filters[1], stage_filters[2], stride=2, dropout_rate=spatial_dropout
        )

        # 3. Global Max Pooling
        # Preserves peak signal intensity (iceberg signature)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # 4. Classifier Head
        # Input dim = Final CNN Channels + 1 (Incidence Angle)
        clf_in_dim = stage_filters[2] + 1

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride, dropout_rate):
        """
        Constructs a ResNet stage with 2 blocks.
        """
        layers = []
        # First block handles stride and channel dimension change
        layers.append(
            SpatialDropoutResBlock(in_channels, out_channels, stride, dropout_rate)
        )
        # Second block maintains dimensions
        layers.append(
            SpatialDropoutResBlock(out_channels, out_channels, 1, dropout_rate)
        )
        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Kaiming initialization for layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Image input (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input (B,)
        Returns:
            torch.Tensor: Logits (B, 1)
        """
        # Backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Late Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # Classification
        x = self.classifier(x)

        return x
