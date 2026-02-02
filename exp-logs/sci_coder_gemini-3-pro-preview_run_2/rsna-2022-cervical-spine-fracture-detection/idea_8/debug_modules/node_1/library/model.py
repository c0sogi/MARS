import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SpatialAttention(nn.Module):
    """
    Computes a spatial attention map from feature maps and returns
    spatially weighted embeddings.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Project features to a single channel attention map
        self.project = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)

        # Compute attention map logits
        # logits: (B, 1, H, W)
        attn_logits = self.project(x)
        attn_map = torch.sigmoid(attn_logits)

        # Spatially weighted sum
        # We multiply features by the attention map and sum over spatial dimensions.
        # This acts as a soft-masking mechanism, where the model learns to
        # sum up features only from relevant (fractured) areas.
        # Output: (B, C)
        features = (x * attn_map).sum(dim=(2, 3))

        return features, attn_logits


class DualAttentionNetwork(nn.Module):
    """
    Idea 8: 2.5D Dual-Attention Network with Spatially-Guided Feature Aggregation.

    Architecture:
    1. 2.5D Backbone (EfficientNet-B4) extracting P4+P5 features.
    2. Spatial Attention Module for intra-slice focus (Supervised).
    3. Bi-LSTM with Positional Embeddings for inter-slice context.
    4. Multi-Head Temporal Attention for study-level classification.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # --- 1. Backbone ---
        # Load EfficientNet-B4, pretrained, extracting intermediate features
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=config.IN_CHANNELS,
        )

        # Dynamically determine feature channels
        # We use a dummy forward pass to get exact channel counts for P4 and P5
        dummy_input = torch.randn(
            1, config.IN_CHANNELS, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]
        )
        with torch.no_grad():
            feats = self.backbone(dummy_input)
            # P4 is usually index 3, P5 is index 4 in features_only output
            p4_ch = feats[3].shape[1]
            p5_ch = feats[4].shape[1]

        self.concat_channels = p4_ch + p5_ch

        # --- 2. Spatial Attention ---
        self.spatial_attention = SpatialAttention(self.concat_channels)

        # --- 3. Sequence Modeling ---
        self.lstm_input_size = self.concat_channels
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0,
        )

        # Learnable Positional Embeddings
        # Shape: (1, Seq_Len, Input_Dim) - Broadcasts across batch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.SEQ_LEN, self.lstm_input_size)
        )
        # Initialize with small random values or zeros (zeros is fine as it's learnable)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- 4. Output Heads ---

        # A. Auxiliary Slice Head (Binary Classification per slice)
        # Input: LSTM output (B, Seq, 2*Hidden)
        self.slice_head = nn.Linear(config.LSTM_HIDDEN_SIZE * 2, 1)

        # B. Primary Study Head (Class-Specific Temporal Attention)
        # We create 8 separate attention heads, one for each target class.
        # This allows the model to attend to different slices for different vertebrae.
        self.num_classes = len(config.TARGET_COLS)

        # Attention mechanism: Linear -> Tanh -> Linear -> Score
        self.temporal_attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.LSTM_HIDDEN_SIZE * 2, 128),
                    nn.Tanh(),
                    nn.Linear(128, 1),
                )
                for _ in range(self.num_classes)
            ]
        )

        # Classifiers: One linear layer per class applied to its specific context vector
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.LSTM_HIDDEN_SIZE * 2, 1) for _ in range(self.num_classes)]
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels, Height, Width)
        Returns:
            Dict containing 'study_logits', 'slice_logits', 'spatial_logits'
        """
        b, seq, c, h, w = x.shape

        # Merge Batch and Sequence dimensions for parallel backbone processing
        # (B*S, C, H, W)
        x = x.view(b * seq, c, h, w)

        # --- Backbone Feature Extraction ---
        features = self.backbone(x)
        p4 = features[3]  # Stride 16 (e.g., 24x24 for 384 input)
        p5 = features[4]  # Stride 32 (e.g., 12x12 for 384 input)

        # Upsample P5 to match P4 spatial dimensions
        p5_up = F.interpolate(
            p5, size=p4.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate features
        # (B*S, C_total, H', W')
        concat_feats = torch.cat([p4, p5_up], dim=1)

        # --- Spatial Attention & Pooling ---
        # slice_embeds: (B*S, C_total)
        # spatial_logits: (B*S, 1, H', W') - Used for auxiliary supervision
        slice_embeds, spatial_logits = self.spatial_attention(concat_feats)

        # --- Sequence Modeling ---
        # Reshape back to sequence format: (B, S, C_total)
        slice_embeds = slice_embeds.view(b, seq, -1)

        # Add Positional Embeddings
        slice_embeds = slice_embeds + self.pos_embed

        # Bi-LSTM
        # lstm_out: (B, S, 2*Hidden)
        lstm_out, _ = self.lstm(slice_embeds)

        # --- Predictions ---

        # 1. Slice Level Auxiliary Prediction
        # (B, S, 1) -> (B, S)
        slice_logits = self.slice_head(lstm_out).squeeze(-1)

        # 2. Study Level Prediction (Temporal Attention)
        study_logits_list = []

        for i in range(self.num_classes):
            # Calculate attention scores for class i
            # (B, S, 2H) -> (B, S, 1)
            att_scores = self.temporal_attentions[i](lstm_out)
            att_weights = torch.softmax(att_scores, dim=1)

            # Weighted Sum (Context Vector)
            # (B, S, 2H) * (B, S, 1) -> Sum over S -> (B, 2H)
            context = (lstm_out * att_weights).sum(dim=1)

            # Classification
            logit = self.classifiers[i](context)
            study_logits_list.append(logit)

        # Concatenate all class logits: (B, 8)
        study_logits = torch.cat(study_logits_list, dim=1)

        # Reshape spatial logits to separate batch and sequence for loss calculation
        # (B, S, 1, H', W')
        spatial_logits = spatial_logits.view(
            b, seq, 1, concat_feats.shape[2], concat_feats.shape[3]
        )

        return {
            "study_logits": study_logits,
            "slice_logits": slice_logits,
            "spatial_logits": spatial_logits,
        }
