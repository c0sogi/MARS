import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_CHANNELS,
    BACKBONE_CHANNELS,
    LEAKY_RELU_SLOPE,
    SE_REDUCTION_RATIO,
    READOUT_PROJ_DIM,
    FEATURE_DIM,
    CALIBRATION_HIDDEN_DIM,
    DROPOUT_RATE,
)


class SEModule(nn.Module):
    """
    Standard Squeeze-and-Excitation Module.
    Structure: GlobalAvgPool -> FC -> ReLU -> FC -> Sigmoid -> Scale
    """

    def __init__(self, channels, reduction=SE_REDUCTION_RATIO):
        super(SEModule, self).__init__()
        # Ensure reduction doesn't make hidden dim 0
        hidden_dim = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CalibrationHead(nn.Module):
    """
    Physics-informed Multiplicative Calibration Head.
    Takes incidence angle and produces Scale (Gain) and Shift (Bias) vectors.
    """

    def __init__(
        self, input_dim=1, hidden_dim=CALIBRATION_HIDDEN_DIM, output_dim=FEATURE_DIM
    ):
        super(CalibrationHead, self).__init__()
        # Output dim is doubled because we generate both Scale and Shift
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True),
            nn.Linear(hidden_dim, output_dim * 2),
        )
        self.output_dim = output_dim

    def forward(self, angle):
        # angle shape: (batch_size, 1) or (batch_size)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Raw output: (batch_size, output_dim * 2)
        raw_out = self.mlp(angle)

        # Split into Scale and Shift
        # scale_raw, shift_raw: (batch_size, output_dim)
        scale_raw, shift_raw = torch.split(raw_out, self.output_dim, dim=1)

        # Apply modulation to Scale: Sigmoid(S) * 2 -> Range [0, 2]
        scale = torch.sigmoid(scale_raw) * 2.0

        # Shift is additive, keep as is (or could be tanh, but linear is standard for bias)
        shift = shift_raw

        return scale, shift


class MCICNN(nn.Module):
    """
    Multiplicative-Calibrated Isomorphic CNN (MCI-CNN).

    Architecture:
    1. 4-Stage Plain CNN Backbone (Conv-BN-Leaky-SE-Pool)
    2. Decoupled Isomorphic Readout (Stage 3 & 4 -> Proj -> Max/Min Pool)
    3. Multiplicative Calibration (Image Feats * Gain + Bias)
    4. Classification Head
    """

    def __init__(self):
        super(MCICNN, self).__init__()

        # --- Backbone ---
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                BACKBONE_CHANNELS[0],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(BACKBONE_CHANNELS[0]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(BACKBONE_CHANNELS[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                BACKBONE_CHANNELS[0],
                BACKBONE_CHANNELS[1],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(BACKBONE_CHANNELS[1]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(BACKBONE_CHANNELS[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(
                BACKBONE_CHANNELS[1],
                BACKBONE_CHANNELS[2],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(BACKBONE_CHANNELS[2]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(BACKBONE_CHANNELS[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            nn.Conv2d(
                BACKBONE_CHANNELS[2],
                BACKBONE_CHANNELS[3],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(BACKBONE_CHANNELS[3]),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            SEModule(BACKBONE_CHANNELS[3]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Readout (Decoupled Isomorphic) ---
        # Projections for Stage 3 and Stage 4
        self.proj3 = nn.Conv2d(
            BACKBONE_CHANNELS[2], READOUT_PROJ_DIM, kernel_size=1, bias=True
        )
        self.proj4 = nn.Conv2d(
            BACKBONE_CHANNELS[3], READOUT_PROJ_DIM, kernel_size=1, bias=True
        )

        # Global Pooling layers
        self.global_max = nn.AdaptiveMaxPool2d(1)
        # Min pooling implemented as -Max(-x)

        # --- Calibration Head ---
        self.calibration = CalibrationHead(
            input_dim=1, hidden_dim=CALIBRATION_HIDDEN_DIM, output_dim=FEATURE_DIM
        )

        # --- Classification Head ---
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.classifier = nn.Linear(FEATURE_DIM, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x: Image tensor (B, 3, 75, 75)
            angle: Incidence angle tensor (B, 1) or (B,)
        """
        # Backbone Forward
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Readout: Stage 3
        p3 = self.proj3(x3)
        max3 = self.global_max(p3).flatten(1)
        min3 = -self.global_max(-p3).flatten(1)

        # Readout: Stage 4
        p4 = self.proj4(x4)
        max4 = self.global_max(p4).flatten(1)
        min4 = -self.global_max(-p4).flatten(1)

        # Concatenate features (B, 256)
        # Order: [Max3, Min3, Max4, Min4] -> 64*4 = 256
        img_features = torch.cat([max3, min3, max4, min4], dim=1)

        # Calibration
        # scale: (B, 256), shift: (B, 256)
        scale, shift = self.calibration(angle)

        # Fusion: Multiplicative Gain + Additive Bias
        calibrated_features = img_features * scale + shift

        # Classification
        out = self.dropout(calibrated_features)
        logits = self.classifier(out)

        return logits
