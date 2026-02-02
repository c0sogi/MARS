import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling Layer.

    Replaces Global Average Pooling. Instead of averaging all spatial locations equally,
    it learns to assign weights to different time-frequency regions based on their content.

    Formula:
        a = softmax( w^T * (tanh(V*x) * sigmoid(U*x)) )
        output = sum(a * x)
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(GatedAttentionPooling, self).__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        batch_size, channels, height, width = x.size()

        # Flatten spatial dimensions: [Batch, Channels, H*W] -> [Batch, H*W, Channels]
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)

        # Gated Attention Mechanism
        # tanh(V*x)
        att_v = torch.tanh(self.V(x_flat))
        # sigmoid(U*x)
        att_u = torch.sigmoid(self.U(x_flat))

        # Element-wise multiplication and projection
        # w^T * (...)
        att_score = self.w(att_v * att_u)  # [Batch, H*W, 1]

        # Softmax over spatial dimension (H*W)
        att_weights = F.softmax(att_score, dim=1)

        # Weighted sum: sum(weights * features)
        # [Batch, H*W, 1] * [Batch, H*W, C] -> [Batch, H*W, C] -> sum -> [Batch, C]
        x_weighted = torch.sum(x_flat * att_weights, dim=1)

        return x_weighted


class AttentionPooledHybridEfficientNet(nn.Module):
    """
    Hybrid architecture combining an EfficientNet-B0 backbone for spectrograms
    and an MLP for statistical features.

    Key Features:
    1. Modified first conv layer for 10-channel input.
    2. Gated Attention Pooling to focus on transient seismic events.
    3. Dual-branch fusion.
    """

    def __init__(self):
        super(AttentionPooledHybridEfficientNet, self).__init__()

        # ---------------------------------------------------------
        # 1. Spectrogram Branch (EfficientNet Backbone)
        # ---------------------------------------------------------
        # Load pretrained EfficientNet-B0
        # num_classes=0 and global_pool='' ensures we get the feature map (N, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0, global_pool=""
        )

        # Modify the first convolution layer to accept 10 channels
        self._modify_first_conv_layer()

        # Determine backbone output channels
        # For EfficientNet-B0, this is typically 1280
        dummy_input = torch.randn(1, Config.IN_CHANNELS, 128, 128)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            backbone_out_channels = features.shape[1]

        # Attention Pooling Layer
        self.attention_pool = GatedAttentionPooling(
            input_dim=backbone_out_channels, hidden_dim=Config.HIDDEN_DIM
        )

        # ---------------------------------------------------------
        # 2. Statistical Branch (MLP)
        # ---------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(Config.NUM_STAT_FEATURES, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # ---------------------------------------------------------
        # 3. Fusion Head
        # ---------------------------------------------------------
        # Concatenation of Backbone Output (C) + MLP Output (Hidden)
        fusion_dim = backbone_out_channels + Config.HIDDEN_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, 1),
        )

    def _modify_first_conv_layer(self):
        """
        Adapts the first convolutional layer of the backbone to accept
        Config.IN_CHANNELS (10) instead of the default 3 (RGB).

        Weights are initialized by averaging the RGB weights and replicating them.
        """
        # Get the existing conv layer (usually named conv_stem in EfficientNet)
        old_conv = self.backbone.conv_stem

        # Create new conv layer with same parameters but different in_channels
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights
        # Shape: [Out, In, K, K]
        with torch.no_grad():
            # Average the weights across the original 3 channels -> [Out, 1, K, K]
            weight_mean = old_conv.weight.mean(dim=1, keepdim=True)
            # Replicate 10 times -> [Out, 10, K, K]
            new_weight = weight_mean.repeat(1, Config.IN_CHANNELS, 1, 1)
            new_conv.weight.copy_(new_weight)

            # Copy bias if it exists
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, spectrogram, features):
        """
        Forward pass of the hybrid model.

        Args:
            spectrogram (torch.Tensor): [Batch, 10, Freq, Time]
            features (torch.Tensor): [Batch, Num_Stats]

        Returns:
            torch.Tensor: Predicted time to eruption (scaled) [Batch, 1]
        """
        # ---------------------------------------------------------
        # Branch 1: Spectrogram
        # ---------------------------------------------------------
        # Extract feature maps: [Batch, 1280, H, W]
        x_spec = self.backbone(spectrogram)

        # Apply Gated Attention Pooling: [Batch, 1280]
        x_spec_emb = self.attention_pool(x_spec)

        # ---------------------------------------------------------
        # Branch 2: Statistical Features
        # ---------------------------------------------------------
        # Process stats: [Batch, Hidden]
        x_stats_emb = self.mlp(features)

        # ---------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------
        # Concatenate: [Batch, 1280 + Hidden]
        x_fused = torch.cat((x_spec_emb, x_stats_emb), dim=1)

        # Final Prediction
        output = self.head(x_fused)

        return output
