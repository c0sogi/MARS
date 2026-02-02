import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.checkpoint import checkpoint
from library.config import Config


class SpatialAttention(nn.Module):
    """
    Generates a spatial attention map from feature maps and performs
    attention-weighted pooling to produce a slice embedding.
    """

    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.project = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B*Seq, C, H, W)

        # 1. Generate Spatial Attention Map (Logits)
        # We return logits for BCE/Dice supervision
        attn_logits = self.conv(x)  # (B*Seq, 1, H, W)

        # 2. Apply Softmax over spatial dimensions to get weights
        b, c, h, w = x.size()
        attn_map = attn_logits.view(b, 1, h * w)
        attn_weights = F.softmax(attn_map, dim=-1)  # (B*Seq, 1, H*W)

        # 3. Weighted Pooling
        x_flat = x.view(b, c, h * w)
        # (B*Seq, C, H*W) * (B*Seq, 1, H*W) -> sum -> (B*Seq, C)
        pooled = torch.sum(x_flat * attn_weights, dim=2)

        # 4. Projection
        embedding = self.project(pooled)

        return embedding, attn_logits


class AttentionPoolingHead(nn.Module):
    """
    Sequence-level attention pooling head for a specific class (e.g., C1).
    Aggregates the sequence of embeddings into a single study-level vector.
    """

    def __init__(self, input_dim):
        super(AttentionPoolingHead, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # Calculate attention weights
        attn_logits = self.attention(x)  # (Batch, Seq, 1)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Weighted sum
        context = torch.sum(x * attn_weights, dim=1)  # (Batch, Input_Dim)

        # Classification
        logits = self.classifier(context)  # (Batch, 1)

        return logits


class CervicalFractureNet(nn.Module):
    """
    Calibrated 2.5D Dual-Attention Network with Anatomical Injection.
    """

    def __init__(self):
        super(CervicalFractureNet, self).__init__()

        # --- 1. Backbone (EfficientNet-B4) ---
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature channels dynamically
        # EfficientNet-B4 usually: P4 (~160/112), P5 (1792)
        # We run a dummy pass to get exact shapes
        dummy_input = torch.randn(1, Config.IN_CHANNELS, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy_input)
            # We use the last two feature maps
            c_p5 = feats[-1].shape[1]
            c_p4 = feats[-2].shape[1]

        self.fused_dim = c_p5 + c_p4

        # --- 2. Spatial Attention Module ---
        self.spatial_attention = SpatialAttention(
            in_channels=self.fused_dim, out_channels=Config.EMBEDDING_DIM
        )

        # --- 3. Slice Fracture Head (Auxiliary) ---
        # Predicts fracture probability per slice directly from embedding
        self.slice_fracture_head = nn.Linear(Config.EMBEDDING_DIM, 1)

        # --- 4. Sequence Modeling (Bi-LSTM) ---
        self.lstm = nn.LSTM(
            input_size=Config.EMBEDDING_DIM,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=0.1 if Config.LSTM_LAYERS > 1 else 0.0,
        )

        lstm_out_dim = (
            Config.LSTM_HIDDEN_SIZE * 2
            if Config.BIDIRECTIONAL
            else Config.LSTM_HIDDEN_SIZE
        )

        # --- 5. Anatomical Injection ---
        # Predicts vertebral level (0-7) for each slice
        self.anatomy_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, Config.ANATOMY_CLASSES),
        )

        # Input dimension for final heads includes the anatomy probabilities
        # We concatenate the softmax output of anatomy head (size 8) to LSTM output
        final_input_dim = lstm_out_dim + Config.ANATOMY_CLASSES

        # --- 6. Classification Heads ---
        # 8 Independent heads: C1, C2, C3, C4, C5, C6, C7, Patient_Overall
        self.heads = nn.ModuleList(
            [AttentionPoolingHead(final_input_dim) for _ in range(Config.NUM_TARGETS)]
        )

    def forward(self, x):
        # x: (Batch, Seq, C, H, W)
        b, s, c, h, w = x.size()

        # Flatten Batch and Sequence for 2D Backbone
        x_flat = x.view(b * s, c, h, w)

        # 1. Feature Extraction (Chunked)
        # Process in chunks to avoid OOM with large sequence lengths
        chunk_size = 16  # Process 16 slices at a time
        p4_list = []
        p5_list = []

        for i in range(0, x_flat.size(0), chunk_size):
            chunk = x_flat[i : i + chunk_size]

            if self.training:
                # Gradient Checkpointing: Process chunk with recomputation
                # We detach and set requires_grad to ensure the hook is triggered
                chunk_input = chunk.detach()
                chunk_input.requires_grad = True

                def run_backbone(c):
                    # timm returns a list, checkpoint expects tuple/tensor
                    return tuple(self.backbone(c))

                features_tuple = checkpoint(
                    run_backbone, chunk_input, use_reentrant=False
                )
                features = list(features_tuple)
            else:
                features = self.backbone(chunk)

            p4_list.append(features[-2])
            p5_list.append(features[-1])

        p4 = torch.cat(p4_list, dim=0)
        p5 = torch.cat(p5_list, dim=0)

        if p5.shape[-2:] != p4.shape[-2:]:
            p5 = F.interpolate(
                p5, size=p4.shape[-2:], mode="bilinear", align_corners=False
            )

        fused_feats = torch.cat([p4, p5], dim=1)  # (B*S, fused_dim, H_feat, W_feat)

        # 2. Spatial Attention & Embedding
        # embedding: (B*S, Embed_Dim)
        # spatial_logits: (B*S, 1, H_feat, W_feat)
        embedding, spatial_logits = self.spatial_attention(fused_feats)

        # 3. Reshape for Sequence Modeling
        # (B, S, Embed_Dim)
        seq_embedding = embedding.view(b, s, -1)

        # Auxiliary: Slice Fracture Prediction
        slice_fracture_logits = self.slice_fracture_head(seq_embedding)  # (B, S, 1)

        # 4. LSTM
        lstm_out, _ = self.lstm(seq_embedding)  # (B, S, LSTM_Out_Dim)

        # 5. Anatomical Injection
        anatomy_logits = self.anatomy_head(lstm_out)  # (B, S, 8)
        anatomy_probs = F.softmax(anatomy_logits, dim=-1)  # (B, S, 8)

        # Concatenate Anatomy Probs to LSTM features
        # This gives the attention heads explicit signal: "This is C1", "This is C2"
        enriched_features = torch.cat(
            [lstm_out, anatomy_probs], dim=-1
        )  # (B, S, LSTM_Out + 8)

        # 6. Classification Heads
        study_logits_list = []
        for head in self.heads:
            # Each head outputs (B, 1)
            study_logits_list.append(head(enriched_features))

        # Concatenate to (B, 8)
        study_logits = torch.cat(study_logits_list, dim=1)

        # Reshape spatial logits for output
        # (B, S, H_feat, W_feat)
        spatial_maps = spatial_logits.view(
            b, s, spatial_logits.size(2), spatial_logits.size(3)
        )

        # Upsample spatial maps to original image size for loss calculation if needed
        # Or we can downsample masks in loss. Usually better to upsample prediction.
        spatial_maps = F.interpolate(
            spatial_maps.view(b * s, 1, spatial_maps.size(2), spatial_maps.size(3)),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).view(b, s, h, w)

        return {
            "study_logits": study_logits,
            "slice_fracture_logits": slice_fracture_logits,
            "spatial_maps": spatial_maps,
            "anatomy_logits": anatomy_logits,
        }
