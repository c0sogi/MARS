import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolution Block: Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationUNet(nn.Module):
    """
    Stage 1: Global Context & Localization Stream.
    Outputs:
        1. Segmentation Map (B, 8, H, W)
        2. Global Context Vector (B, 512)
        3. Anatomical Probability Map (B, 8)
    """

    def __init__(self):
        super().__init__()

        # Encoder: ResNet18
        # Features: [C1, C2, C3, C4, C5] corresponding to strides [2, 4, 8, 16, 32]
        # Channels: [64, 64, 128, 256, 512]
        self.encoder = timm.create_model(
            Config.STAGE1_BACKBONE,
            features_only=True,
            pretrained=True,
            in_chans=Config.STAGE1_IN_CHANNELS,
        )

        enc_channels = self.encoder.feature_info.channels()

        # Bottleneck Context Dimension
        self.context_dim = enc_channels[-1]

        # Decoder
        # We upsample from the bottom up
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(enc_channels[-1] + enc_channels[-2], 256)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(256 + enc_channels[-3], 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(128 + enc_channels[-4], 64)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = ConvBlock(64 + enc_channels[-5], 64)

        # Final upsample to match input resolution (stride 2 -> 1)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec5 = ConvBlock(64, 32)

        # Segmentation Head
        self.seg_head = nn.Conv2d(32, Config.STAGE1_NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features[0]: stride 2, 64 ch
        # features[1]: stride 4, 64 ch
        # features[2]: stride 8, 128 ch
        # features[3]: stride 16, 256 ch
        # features[4]: stride 32, 512 ch

        # 1. Global Context Vector
        # Global Average Pooling on the deepest feature map
        global_context = F.adaptive_avg_pool2d(features[-1], (1, 1)).flatten(1)

        # Decoder
        x_dec = self.up1(features[-1])
        x_dec = torch.cat([x_dec, features[-2]], dim=1)
        x_dec = self.dec1(x_dec)

        x_dec = self.up2(x_dec)
        x_dec = torch.cat([x_dec, features[-3]], dim=1)
        x_dec = self.dec2(x_dec)

        x_dec = self.up3(x_dec)
        x_dec = torch.cat([x_dec, features[-4]], dim=1)
        x_dec = self.dec3(x_dec)

        x_dec = self.up4(x_dec)
        x_dec = torch.cat([x_dec, features[-5]], dim=1)
        x_dec = self.dec4(x_dec)

        x_dec = self.up5(x_dec)
        x_dec = self.dec5(x_dec)

        # 2. Segmentation Logits
        seg_logits = self.seg_head(x_dec)

        # 3. Anatomical Probability Map
        # Softmax over classes (dim 1), then average over spatial dimensions (dim 2,3)
        # This gives the average probability of each class being present in the slice
        seg_probs_spatial = F.softmax(seg_logits, dim=1)
        anatomical_probs = F.adaptive_avg_pool2d(seg_probs_spatial, (1, 1)).flatten(1)

        return seg_logits, global_context, anatomical_probs


class FractureEncoder(nn.Module):
    """
    Stage 2: Local Fracture Stream (2.5D CNN).
    Input: (B, 4, H, W) - 3 slices + 1 mask
    Output: Local Fracture Embedding (B, EmbedDim)
    """

    def __init__(self):
        super().__init__()

        # Backbone: EfficientNet-V2
        # in_chans=4 adapts the first layer weights automatically
        self.backbone = timm.create_model(
            Config.STAGE2_BACKBONE,
            pretrained=True,
            num_classes=0,  # Remove classifier head
            in_chans=Config.STAGE2_IN_CHANNELS,
        )

        # Get output feature dimension
        dummy_input = torch.randn(1, Config.STAGE2_IN_CHANNELS, 256, 256)
        with torch.no_grad():
            out_dim = self.backbone(dummy_input).shape[1]

        # Projection head to ensure consistent embedding size
        self.projection = nn.Sequential(
            nn.Linear(out_dim, Config.STAGE2_EMBEDDING_DIM),
            nn.LayerNorm(Config.STAGE2_EMBEDDING_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projection(features)
        return embedding


class AttentionHead(nn.Module):
    """
    Standard Attention Mechanism for aggregating sequence data.
    Used for the 'patient_overall' head.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, SeqLen, HiddenDim)
        attn_weights = self.attention(x)  # (B, SeqLen, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(x * attn_weights, dim=1)  # (B, HiddenDim)
        return context


class DualStreamRNN(nn.Module):
    """
    Stage 3: Dual-Stream Soft-Attention Aggregator.
    Inputs:
        - Local Fracture Embeddings (Sequence)
        - Global Context Vectors (Sequence)
        - Anatomical Probabilities (Sequence)
    Output:
        - 8 probabilities (C1-C7, Patient Overall)
    """

    def __init__(self, global_context_dim=512):
        super().__init__()

        # Input Dimension
        # Local Embedding (512) + Global Context (512 usually) + Anatomical Probs (8)
        self.input_dim = (
            Config.STAGE2_EMBEDDING_DIM + global_context_dim + Config.STAGE1_NUM_CLASSES
        )

        # Bi-Directional GRU
        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=Config.STAGE3_RNN_HIDDEN_SIZE,
            num_layers=Config.STAGE3_RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.STAGE3_DROPOUT if Config.STAGE3_RNN_LAYERS > 1 else 0,
        )

        self.rnn_out_dim = Config.STAGE3_RNN_HIDDEN_SIZE * 2

        # Patient Overall Attention Head
        self.patient_attention = AttentionHead(self.rnn_out_dim)
        self.patient_classifier = nn.Linear(self.rnn_out_dim, 1)

        # Vertebrae Classifiers (C1-C7)
        # We share the linear layer or have separate ones?
        # Separate is better to allow specific feature extraction per level.
        self.vertebrae_classifiers = nn.ModuleList(
            [nn.Linear(self.rnn_out_dim, 1) for _ in range(7)]
        )

    def forward(self, local_emb, global_ctx, anat_probs):
        """
        Args:
            local_emb: (B, SeqLen, 512)
            global_ctx: (B, SeqLen, 512)
            anat_probs: (B, SeqLen, 8) - Index 0 is background, 1-7 are C1-C7
        """
        # 1. Concatenate Inputs
        # (B, SeqLen, 1032)
        rnn_input = torch.cat([local_emb, global_ctx, anat_probs], dim=2)

        # 2. Sequence Modeling
        # rnn_out: (B, SeqLen, Hidden*2)
        rnn_out, _ = self.rnn(rnn_input)

        outputs = []

        # 3. Specific Vertebrae Heads (C1-C7)
        # anat_probs indices: 0=Background, 1=C1, ..., 7=C7
        for i in range(7):
            vert_idx = i + 1  # C1 is at index 1

            # Get probabilities for this vertebra across the sequence
            # (B, SeqLen, 1)
            p_vert = anat_probs[:, :, vert_idx].unsqueeze(-1)

            # Normalize to create attention weights
            # Add epsilon to avoid division by zero if vertebra not present
            weights = p_vert / (torch.sum(p_vert, dim=1, keepdim=True) + 1e-6)

            # Weighted aggregation of RNN states
            # (B, Hidden*2)
            context = torch.sum(rnn_out * weights, dim=1)

            # Classification
            logits = self.vertebrae_classifiers[i](context)
            outputs.append(logits)

        # 4. Patient Overall Head
        # Uses learned self-attention over the whole sequence
        patient_context = self.patient_attention(rnn_out)
        patient_logits = self.patient_classifier(patient_context)
        outputs.append(patient_logits)

        # Stack outputs: [C1, C2, ..., C7, Patient]
        # (B, 8)
        final_logits = torch.cat(outputs, dim=1)

        return final_logits
