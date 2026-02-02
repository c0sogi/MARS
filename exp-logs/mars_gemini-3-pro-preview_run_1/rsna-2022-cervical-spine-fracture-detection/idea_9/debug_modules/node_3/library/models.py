import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config

# =============================================================================
# Utility Blocks
# =============================================================================


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle slight shape mismatches due to padding in encoder
            if x.shape[2:] != skip.shape[2:]:
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


class SoftAnatomicalPooling(nn.Module):
    """
    Attention mechanism to aggregate RNN hidden states into vertebra-specific context vectors.
    Uses the 'Soft Anatomical Map' (probabilities from Stage 1) as a prior for attention.
    """

    def __init__(self, hidden_size, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Learnable projection for attention score calculation
        self.attn_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, anatomical_map):
        """
        Args:
            hidden_states: (Batch, Seq_Len, Hidden_Dim)
            anatomical_map: (Batch, Seq_Len, Num_Classes) - Probabilities of C1-C7 presence per slice

        Returns:
            context_vectors: (Batch, Num_Classes, Hidden_Dim)
        """
        # 1. Calculate raw content-based attention scores: (B, Seq, 7)
        attn_scores = self.attn_proj(hidden_states)

        # 2. Incorporate Anatomical Prior
        # We add log(prob) to the scores to bias attention towards slices where the vertebra is present.
        # Add epsilon for numerical stability.
        prior_log = torch.log(anatomical_map + 1e-6)

        # 3. Compute Attention Weights
        # Softmax over the Sequence dimension (dim=1) to normalize weights for each class
        combined_scores = attn_scores + prior_log
        attn_weights = F.softmax(combined_scores, dim=1)  # (B, Seq, 7)

        # 4. Weighted Aggregation
        # hidden: (B, Seq, Hidden) -> (B, 1, Seq, Hidden)
        # weights: (B, Seq, 7) -> (B, 7, Seq, 1)
        h_expanded = hidden_states.unsqueeze(1)
        w_expanded = attn_weights.permute(0, 2, 1).unsqueeze(-1)

        # Sum over sequence dimension: (B, 7, Hidden)
        context_vectors = torch.sum(h_expanded * w_expanded, dim=2)

        return context_vectors


# =============================================================================
# Stage 1: Anatomical Localizer
# =============================================================================


class AnatomicalLocalizer(nn.Module):
    """
    2D U-Net for Segmentation and Anatomical Probability Estimation.
    Backbone: ResNet18 (timm)
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Encoder: ResNet18
        # Features: [C1(64), C2(64), C3(128), C4(256), C5(512)]
        # Strides:  [2,      4,      8,       16,      32]
        self.encoder = timm.create_model(
            Config.SEG_BACKBONE,
            features_only=True,
            pretrained=pretrained,
            in_chans=Config.SEG_IN_CHANS,
        )

        # Center Block
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder Path
        # Dec5: In 512 + Skip 256 (C4) -> Out 256
        self.dec5 = DecoderBlock(512, 256, 256)
        # Dec4: In 256 + Skip 128 (C3) -> Out 128
        self.dec4 = DecoderBlock(256, 128, 128)
        # Dec3: In 128 + Skip 64 (C2) -> Out 64
        self.dec3 = DecoderBlock(128, 64, 64)
        # Dec2: In 64 + Skip 64 (C1) -> Out 32
        self.dec2 = DecoderBlock(64, 64, 32)
        # Dec1: In 32 + Skip 0 -> Out 16 (Final Upsample to original res)
        self.dec1 = DecoderBlock(32, 0, 16)

        # Final Segmentation Head
        self.final_conv = nn.Conv2d(16, Config.SEG_NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, H, W)

        # Encoder
        features = self.encoder(x)
        e1, e2, e3, e4, e5 = features

        # Center
        c = self.center(e5)

        # Decoder
        d5 = self.dec5(c, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2)  # Upsamples to original H, W

        # Segmentation Logits: (B, 8, H, W)
        mask_logits = self.final_conv(d1)

        # Soft Anatomical Map (Presence Probabilities)
        # Extract channels 1-7 (C1-C7), ignore background (0)
        c_logits = mask_logits[:, 1:, :, :]  # (B, 7, H, W)

        # Global Max Pooling to detect if vertebra exists anywhere in slice
        # (B, 7, H*W) -> max -> (B, 7)
        presence_logits = F.adaptive_max_pool2d(c_logits, (1, 1)).view(
            c_logits.size(0), 7
        )
        presence_probs = torch.sigmoid(presence_logits)

        return mask_logits, presence_probs


# =============================================================================
# Stage 2: Dual-Branch Feature Encoder
# =============================================================================


class DualBranchEncoder(nn.Module):
    """
    Dual-Stream CNN for Slice Feature Extraction.
    Branch 1 (Local): High-Res Crop + Bone Mask (4 channels).
    Branch 2 (Global): Downsampled Full Slice (3 channels).
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Global Branch (EfficientNet-B0)
        self.global_enc = timm.create_model(
            Config.ENC_BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.GLOBAL_IN_CHANS,
        )

        # Local Branch (EfficientNet-B0)
        # in_chans=4: timm automatically adapts weights (e.g., repeating/averaging)
        self.local_enc = timm.create_model(
            Config.ENC_BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.LOCAL_IN_CHANS,
        )

        # Determine feature dimension (usually 1280 for B0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.global_enc(dummy).shape[1]

        # Fusion Layer
        # Concatenates both vectors and projects to embedding dim
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, Config.ENC_EMBED_DIM),
            nn.BatchNorm1d(Config.ENC_EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Auxiliary Head for Pre-training (Slice-level Fracture Binary Classification)
        self.head = nn.Linear(Config.ENC_EMBED_DIM, 1)

    def forward(self, local_inputs, global_inputs):
        # local_inputs: (B, 4, H, W)
        # global_inputs: (B, 3, H, W)

        local_feat = self.local_enc(local_inputs)
        global_feat = self.global_enc(global_inputs)

        # Concatenate
        concat = torch.cat([local_feat, global_feat], dim=1)

        # Fuse
        embedding = self.fusion(concat)

        # Auxiliary prediction
        logits = self.head(embedding)

        return embedding, logits


# =============================================================================
# Stage 3: Hierarchical Aggregator
# =============================================================================


class HierarchicalAggregator(nn.Module):
    """
    Bi-GRU with Soft Anatomical Pooling for Patient-Level Prediction.
    """

    def __init__(self):
        super().__init__()

        # Bi-Directional GRU
        self.gru = nn.GRU(
            input_size=Config.RNN_INPUT_SIZE,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # Hidden dimension doubles for bidirectional
        self.hidden_dim = (
            Config.RNN_HIDDEN_SIZE * 2
            if Config.RNN_BIDIRECTIONAL
            else Config.RNN_HIDDEN_SIZE
        )

        # Anatomical Pooling Module
        self.pooling = SoftAnatomicalPooling(self.hidden_dim, Config.NUM_VERTEBRAE)

        # 1. Vertebrae-Specific Heads (C1-C7)
        # Each head predicts fracture probability for one vertebra using its specific context vector
        self.vert_heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(Config.NUM_VERTEBRAE)]
        )

        # 2. Patient-Overall Head
        # Predicts 'patient_overall' using the concatenation of all context vectors
        self.patient_head = nn.Linear(Config.NUM_VERTEBRAE * self.hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, Seq_Len, Input_Size)
               Input_Size = ENC_EMBED_DIM + 7 (Anatomical Map)
        """
        # Split input into Features and Anatomical Map
        features = x[:, :, : Config.ENC_EMBED_DIM]
        anatomical_map = x[:, :, Config.ENC_EMBED_DIM :]

        # RNN Forward Pass
        # rnn_out: (B, Seq, Hidden*Dirs)
        rnn_out, _ = self.gru(x)

        # Soft Anatomical Pooling
        # context_vectors: (B, 7, Hidden*Dirs)
        context_vectors = self.pooling(rnn_out, anatomical_map)

        # Generate Vertebrae Predictions
        vert_logits_list = []
        for k in range(Config.NUM_VERTEBRAE):
            # Select context vector for vertebra k
            vec = context_vectors[:, k, :]  # (B, Hidden)
            logit = self.vert_heads[k](vec)  # (B, 1)
            vert_logits_list.append(logit)

        # Concatenate C1-C7 logits: (B, 7)
        vert_logits = torch.cat(vert_logits_list, dim=1)

        # Generate Patient Prediction
        # Flatten all context vectors: (B, 7 * Hidden)
        patient_feat = context_vectors.view(context_vectors.size(0), -1)
        patient_logit = self.patient_head(patient_feat)  # (B, 1)

        # Final Output: Concatenate [C1..C7, Patient] -> (B, 8)
        all_logits = torch.cat([vert_logits, patient_logit], dim=1)

        return all_logits
