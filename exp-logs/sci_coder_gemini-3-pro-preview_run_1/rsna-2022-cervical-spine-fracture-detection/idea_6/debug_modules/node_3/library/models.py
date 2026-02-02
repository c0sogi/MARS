import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library import config

# ====================================================
# STAGE 1: Multi-Class Anatomical Localizer (2D U-Net)
# ====================================================


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Upsample x
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate with skip connection if provided
        if skip is not None:
            # Handle potential shape mismatch due to padding/rounding
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetLocalizer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder_name = config.STAGE1_CONFIG["model_name"]
        self.num_classes = config.STAGE1_CONFIG["num_classes"]

        # Load encoder with features_only=True to get intermediate feature maps
        # efficientnet_b0 features: [C1, C2, C3, C4, C5]
        # Strides: [2, 4, 8, 16, 32]
        # Channels (approx): [16, 24, 40, 112, 320]
        self.encoder = timm.create_model(
            self.encoder_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get channel counts from the encoder
        feature_info = self.encoder.feature_info.info
        encoder_channels = [info["num_chs"] for info in feature_info]
        # e.g., [16, 24, 40, 112, 320] for effnet_b0

        # Decoder parameters
        decoder_channels = [256, 128, 64, 32, 16]

        # Center block (bottleneck) - no upsampling yet, just processing the deepest feature
        self.center = nn.Sequential(
            nn.Conv2d(
                encoder_channels[-1],
                encoder_channels[-1],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks
        # Block 4: Input (Deepest), Skip (Enc 3) -> Out 256
        self.dec4 = UNetDecoderBlock(
            encoder_channels[-1], encoder_channels[-2], decoder_channels[0]
        )
        # Block 3: Input (Dec 4), Skip (Enc 2) -> Out 128
        self.dec3 = UNetDecoderBlock(
            decoder_channels[0], encoder_channels[-3], decoder_channels[1]
        )
        # Block 2: Input (Dec 3), Skip (Enc 1) -> Out 64
        self.dec2 = UNetDecoderBlock(
            decoder_channels[1], encoder_channels[-4], decoder_channels[2]
        )
        # Block 1: Input (Dec 2), Skip (Enc 0) -> Out 32
        self.dec1 = UNetDecoderBlock(
            decoder_channels[2], encoder_channels[0], decoder_channels[3]
        )

        # Final upsampling block to restore original resolution (Stride 2 -> 1)
        self.final_conv = nn.Sequential(
            UNetDecoderBlock(decoder_channels[3], 0, decoder_channels[4]),
            nn.Conv2d(decoder_channels[4], self.num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x shape: (B, 3, H, W)

        # Encoder pass
        features = self.encoder(x)
        # features[0]: stride 2
        # features[1]: stride 4
        # features[2]: stride 8
        # features[3]: stride 16
        # features[4]: stride 32

        e0, e1, e2, e3, e4 = features

        # Center
        c = self.center(e4)

        # Decoder pass
        # Note: We skip e4 in skip connection for first block, or use e3.
        # Standard U-Net aligns depths.
        # e4 is stride 32. e3 is stride 16.
        # We upsample e4 (now c) to stride 16 and concat with e3.
        d4 = self.dec4(c, e3)  # Stride 16
        d3 = self.dec3(d4, e2)  # Stride 8
        d2 = self.dec2(d3, e1)  # Stride 4
        d1 = self.dec1(d2, e0)  # Stride 2

        # Final upsample to Stride 1
        masks = self.final_conv(d1)  # Stride 1

        # Ensure output size matches input size exactly
        if masks.shape[2:] != x.shape[2:]:
            masks = F.interpolate(
                masks, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        return masks


# ====================================================
# STAGE 2: Mask-Conditioned Feature Encoder (2.5D CNN)
# ====================================================


class MaskedCNNEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone_name = config.STAGE2_CONFIG["backbone"]
        self.in_channels = config.STAGE2_CONFIG["in_channels"]  # 4
        self.feature_dim = config.STAGE2_CONFIG["feature_dim"]  # 1280

        # Load backbone with num_classes=0 to get pooling layer output
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Modify the first convolution layer to accept 4 channels
        # Typically named 'conv_stem' in EfficientNets
        if hasattr(self.backbone, "conv_stem"):
            old_conv = self.backbone.conv_stem
            new_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Initialize weights
            # Copy weights for the first 3 channels (RGB)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Initialize the 4th channel (Mask) to zero or small random values
                # Zero initialization allows the model to start as if the mask doesn't exist
                nn.init.constant_(new_conv.weight[:, 3:, :, :], 0.0)

            self.backbone.conv_stem = new_conv
        else:
            # Fallback for other architectures if needed, though config specifies effnetv2
            raise AttributeError(
                f"Backbone {self.backbone_name} does not have 'conv_stem'. Check layer names."
            )

    def forward(self, x):
        # x shape: (B, 4, H, W)
        features = self.backbone(x)
        # features shape: (B, 1280)
        return features


# ====================================================
# STAGE 3: Anatomically-Grouped Recurrent Aggregator
# ====================================================


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, Seq, Dim)
        # mask: (B, Seq) - 1 for valid, 0 for padding/invalid

        scores = self.attention(x)  # (B, Seq, 1)

        if mask is not None:
            # Set scores for masked steps to -inf so softmax makes them 0
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        weights = F.softmax(scores, dim=1)  # (B, Seq, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, Dim)
        return pooled


class AnatomicalBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = config.STAGE3_CONFIG["hidden_dim"]
        self.num_layers = config.STAGE3_CONFIG["num_layers"]
        self.input_dim = config.STAGE3_CONFIG["input_dim"]  # 1280 + 7
        self.dropout = config.STAGE3_CONFIG["dropout"]
        self.num_vertebrae = config.NUM_VERTEBRAE  # 7

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        self.gru_out_dim = self.hidden_dim * 2

        # Global Attention for Patient Overall
        self.global_attention = AttentionPooling(self.gru_out_dim)
        self.overall_classifier = nn.Linear(self.gru_out_dim, 1)

        # Classifiers for each vertebra
        # We share the classifier weights or keep them separate?
        # Separate is better as C1 fractures might look different from C7.
        self.vert_classifiers = nn.ModuleList(
            [nn.Linear(self.gru_out_dim, 1) for _ in range(self.num_vertebrae)]
        )

    def forward(self, features, anat_ids):
        """
        Args:
            features: (Batch, Seq, FeatureDim) - Visual features from Stage 2
            anat_ids: (Batch, Seq, NumVerts) - One-hot anatomical IDs from Stage 1
        Returns:
            logits: (Batch, 8) - [C1, C2, C3, C4, C5, C6, C7, Overall]
        """
        # Concatenate features and anatomical IDs
        x = torch.cat([features, anat_ids], dim=2)  # (B, Seq, InputDim)

        # Pass through GRU
        gru_out, _ = self.gru(x)  # (B, Seq, Hidden*2)

        batch_size = gru_out.size(0)
        outputs = []

        # 1. Vertebrae Predictions (C1-C7) via Anatomical Pooling
        for k in range(self.num_vertebrae):
            # Get mask for this vertebra: (B, Seq)
            mask_k = anat_ids[:, :, k]

            # We want to pool hidden states where mask_k is active.
            # Using Mean Pooling for robustness, weighted by the mask.

            # Expand mask to hidden dim
            mask_expanded = mask_k.unsqueeze(-1)  # (B, Seq, 1)

            # Sum of hidden states for this vertebra
            masked_sum = torch.sum(gru_out * mask_expanded, dim=1)  # (B, Hidden*2)

            # Count of slices for this vertebra
            mask_count = torch.sum(mask_expanded, dim=1)  # (B, 1)

            # Avoid division by zero (epsilon)
            pooled_k = masked_sum / (mask_count + 1e-6)

            # If a vertebra is not present at all in the sequence (count approx 0),
            # the pooled vector will be 0. The bias in Linear will handle the base probability.

            logits_k = self.vert_classifiers[k](pooled_k)  # (B, 1)
            outputs.append(logits_k)

        # 2. Patient Overall Prediction via Global Attention
        # We use all time steps, but we can mask padding if we had a padding mask.
        # Assuming features are padded with 0 and we don't have explicit length mask passed here,
        # but usually attention handles it if we provided mask.
        # For simplicity, we attend to everything.
        pooled_global = self.global_attention(gru_out)  # (B, Hidden*2)
        logits_overall = self.overall_classifier(pooled_global)  # (B, 1)

        outputs.append(logits_overall)

        # Concatenate all logits: [C1, ..., C7, Overall]
        final_logits = torch.cat(outputs, dim=1)  # (B, 8)

        return final_logits
