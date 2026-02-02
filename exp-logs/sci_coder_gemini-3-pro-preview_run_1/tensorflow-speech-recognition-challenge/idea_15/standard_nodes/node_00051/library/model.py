import torch
import torch.nn as nn
import timm
from library.config import MODEL_PARAMS


class AttentivePooling(nn.Module):
    """
    Attentive Pooling module to dynamically weight active speech segments.
    Replaces standard Global Average Pooling.

    Mechanism:
    1. Project features to a hidden space.
    2. Apply non-linearity (Tanh).
    3. Project to 1 channel to generate attention scores.
    4. Apply Softmax over spatial dimensions (H, W).
    5. Compute weighted sum of features.
    """

    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 8
            # Ensure at least 1 hidden channel
            hidden_channels = max(1, hidden_channels)

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        B, C, H, W = x.size()

        # Calculate attention scores: [B, 1, H, W]
        attn_scores = self.attention(x)

        # Flatten spatial dimensions for softmax: [B, 1, H*W]
        attn_scores = attn_scores.view(B, 1, -1)

        # Calculate attention weights: [B, 1, H*W]
        attn_weights = self.softmax(attn_scores)

        # Flatten input features: [B, C, H*W]
        x_flat = x.view(B, C, -1)

        # Weighted sum: [B, C, H*W] * [B, 1, H*W]^T -> [B, C, 1]
        # We permute weights to [B, H*W, 1] for matrix multiplication
        out = torch.bmm(x_flat, attn_weights.permute(0, 2, 1))

        # Remove last dimension: [B, C]
        out = out.squeeze(-1)

        return out


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling.

    Key Features:
    - Backbone: EfficientNet-B2 initialized with ImageNet weights.
    - Input: 1-channel Spectrogram (weights averaged from RGB).
    - Resolution: Dilated Convolutions (stride=1, dilation=2) in final stage.
    - Head: Attentive Pooling + Linear Classification.
    """

    def __init__(self, config=MODEL_PARAMS):
        super().__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        self.in_channels = config["in_channels"]
        self.dropout_rate = config["dropout"]
        self.pretrained = config["pretrained"]
        self.use_dilated = config["use_dilated_conv"]

        # Determine output stride
        # Standard EfficientNet has output_stride=32 (5 downsampling stages).
        # Setting output_stride=8 forces the last two stages to use dilation,
        # preserving feature resolution (Cite solution_lesson_node_00025).
        output_stride = 8 if self.use_dilated else 32

        # Load Backbone
        # We load with in_chans=3 initially to get standard pretrained weights,
        # then manually patch the stem to ensure correct weight averaging.
        self.backbone = timm.create_model(
            config["model_name"],
            pretrained=self.pretrained,
            output_stride=output_stride,
            in_chans=3,
            features_only=False,
        )

        # Patch First Layer (Input Adaptation)
        if self.in_channels != 3:
            self._adapt_input_conv(self.in_channels)

        # Get feature dimension (EfficientNet-B2: 1408)
        self.num_features = self.backbone.num_features

        # Head Components
        self.att_pool = AttentivePooling(self.num_features)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def _adapt_input_conv(self, in_channels):
        """
        Adapts the first convolutional layer (stem) to accept `in_channels`.
        Weights are initialized by averaging the pretrained RGB weights.
        """
        old_stem = self.backbone.conv_stem

        # Create new stem with correct input channels
        new_stem = nn.Conv2d(
            in_channels,
            old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=(old_stem.bias is not None),
        )

        # Initialize weights
        if self.pretrained and in_channels == 1:
            with torch.no_grad():
                # old_stem.weight shape: [Out, 3, K, K]
                # Average over channel dimension (dim=1)
                avg_weight = old_stem.weight.mean(dim=1, keepdim=True)
                new_stem.weight.copy_(avg_weight)

        # Replace the layer
        self.backbone.conv_stem = new_stem

    def forward(self, x):
        # x: [Batch, 1, Height, Width]

        # Extract features from backbone
        # Output shape: [Batch, 1408, H/16, W/16] (due to output_stride=16)
        x = self.backbone.forward_features(x)

        # Apply Attentive Pooling
        # Output shape: [Batch, 1408]
        x = self.att_pool(x)

        # Classification Head
        x = self.dropout(x)
        logits = self.fc(x)

        return logits
