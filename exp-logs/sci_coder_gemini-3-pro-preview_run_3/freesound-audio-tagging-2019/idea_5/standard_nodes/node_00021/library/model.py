import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class DualStreamPooling(nn.Module):
    """
    Dual-Stream Pooling Head:
    1. Stream A: Non-Linear Attention Pooling (Linear -> Tanh -> Linear -> Softmax)
    2. Stream B: Global Max Pooling
    Fusion: Concatenation of both stream outputs.
    """

    def __init__(self, in_channels, hidden_channels=None):
        super(DualStreamPooling, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 2

        # Attention Mechanism Components
        # Projects feature vectors to a scalar score via a non-linear bottleneck
        self.att_v = nn.Linear(in_channels, hidden_channels)
        self.att_u = nn.Linear(hidden_channels, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (Batch, Channels, Freq, Time)
        B, C, F, T = x.size()

        # Flatten spatial/spectral dimensions: (B, C, N) where N = F * T
        x_flat = x.view(B, C, -1)

        # Permute to (B, N, C) for Linear layers
        x_perm = x_flat.permute(0, 2, 1)

        # --- Stream A: Attention Pooling ---
        # Calculate scores: (B, N, C) -> (B, N, Hidden) -> (B, N, 1)
        # Non-Linear Scoring: Linear -> Tanh -> Linear
        a = self.tanh(self.att_v(x_perm))
        scores = self.att_u(a)

        # Normalize scores over the sequence dimension N to get attention weights
        weights = self.softmax(scores)  # (B, N, 1)

        # Weighted Sum: \sum (weights * features)
        # (B, 1, N) @ (B, N, C) -> (B, 1, C) -> squeeze -> (B, C)
        att_out = torch.bmm(weights.transpose(1, 2), x_perm).squeeze(1)

        # --- Stream B: Global Max Pooling ---
        # Max over N dimension of (B, C, N)
        # This preserves the magnitude of sparse, high-intensity events
        max_out, _ = torch.max(x_flat, dim=2)  # (B, C)

        # --- Fusion ---
        # Concatenate embeddings from both streams
        out = torch.cat([att_out, max_out], dim=1)  # (B, 2*C)

        return out


class AudioEfficientNet(nn.Module):
    def __init__(self):
        super(AudioEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        # Using default (ImageNet) weights
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Input Adaptation (3 Channels -> 1 Channel)
        # Retrieve original first layer
        original_conv = self.backbone.features[0][0]

        # Create new 1-channel convolution with same parameters
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Sum RGB weights to initialize the single channel
        # This preserves the learned filters' intensity response
        with torch.no_grad():
            new_conv.weight[:] = torch.sum(original_conv.weight, dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias[:] = original_conv.bias

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 3. Determine Feature Dimension
        # EfficientNet-B0 typically outputs 1280 channels at the final feature map
        backbone_out_channels = self.backbone.classifier[1].in_features

        # Remove original head (avgpool and classifier) to save memory/compute
        self.backbone.avgpool = nn.Identity()
        self.backbone.classifier = nn.Identity()

        # 4. Dual-Stream Pooling Head
        self.pooling = DualStreamPooling(backbone_out_channels)

        # 5. Classifier
        # Input dimension is doubled due to concatenation of Att and Max streams
        self.fc = nn.Linear(backbone_out_channels * 2, Config.NUM_CLASSES)

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)

        # Extract features using the backbone
        # Output shape: (Batch, 1280, F', T')
        x = self.backbone.features(x)

        # Apply Dual-Stream Pooling
        # Output shape: (Batch, 2560)
        x = self.pooling(x)

        # Final Classification
        # Output shape: (Batch, Num_Classes)
        x = self.fc(x)

        return x
