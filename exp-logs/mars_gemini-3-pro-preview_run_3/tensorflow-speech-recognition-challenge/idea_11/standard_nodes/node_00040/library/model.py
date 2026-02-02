import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import model_cfg


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling layer.
    Aggregates a temporal sequence into a single vector by computing attention scores
    for multiple heads, allowing the model to focus on distinct temporal events.
    """

    def __init__(self, input_dim, num_heads, num_classes, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Project input to calculate attention scores for each head
        # Output shape: (B, T, num_heads)
        self.score_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

        # Final classifier
        # Input: Concatenation of context vectors from all heads
        self.classifier = nn.Linear(input_dim * num_heads, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # Calculate attention scores: (B, T, H)
        scores = self.score_proj(x)

        # Apply Softmax over time dimension to get weights
        attn_weights = F.softmax(scores, dim=1)  # (B, T, H)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum for each head
        # We perform a batch matrix multiplication:
        # Transpose weights to (B, H, T)
        # x is (B, T, D)
        # Result (B, H, D) = (B, H, T) @ (B, T, D)
        attn_weights_t = attn_weights.transpose(1, 2)
        context = torch.bmm(attn_weights_t, x)  # (B, H, D)

        # Flatten heads: (B, H*D)
        context = context.view(B, -1)

        # Classification
        logits = self.classifier(context)
        return logits


class MultiScaleHierarchicalSKResNet(nn.Module):
    """
    Multi-Scale Hierarchical SK-ResNet-CRNN.

    Features:
    1. SK-ResNet34 Backbone with Multi-Resolution Input.
    2. Hierarchical Feature Extraction (Layers 2, 3, 4).
    3. Modified Strides in deep layers to preserve temporal resolution.
    4. BiGRU Neck.
    5. Multi-Head Attention Pooling Head.
    """

    def __init__(self, config=model_cfg):
        super().__init__()
        self.cfg = config

        # 1. Backbone
        # Use features_only=True to extract intermediate maps
        # out_indices=(2, 3, 4) corresponds to the outputs of Layer 2, 3, and 4
        self.backbone = timm.create_model(
            self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=3,
        )

        # 2. Adapt Input Weights
        # Sum RGB weights and replicate to ensure symmetric initialization for multi-res inputs
        if hasattr(self.backbone, "conv1"):
            old_weights = self.backbone.conv1.weight.data  # Shape: (Out, 3, K, K)
            new_weights = old_weights.sum(dim=1, keepdim=True) / 3.0  # Average
            self.backbone.conv1.weight.data = new_weights.repeat(1, 3, 1, 1)

        # 3. Modify Strides
        # Set strides of Layer 3 and Layer 4 to (1, 1) to align temporal resolution with Layer 2
        # We target the first block of each layer where downsampling typically occurs
        self._set_stride_to_one(self.backbone.layer3[0])
        self._set_stride_to_one(self.backbone.layer4[0])

        # 4. Projection Layers (1x1 Conv)
        # Project feature maps to a common dimension before concatenation
        # ResNet34 channel sizes: L2=128, L3=256, L4=512
        self.proj_dim = 128
        self.proj_l2 = nn.Conv1d(128, self.proj_dim, kernel_size=1)
        self.proj_l3 = nn.Conv1d(256, self.proj_dim, kernel_size=1)
        self.proj_l4 = nn.Conv1d(512, self.proj_dim, kernel_size=1)

        # 5. Recurrent Neck (BiGRU)
        # Input size is sum of projected dimensions: 128 * 3 = 384
        gru_input_size = self.proj_dim * 3
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.cfg.gru_hidden_size,
            num_layers=self.cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.gru_dropout if self.cfg.gru_layers > 1 else 0,
        )

        # 6. Attention Head
        gru_out_dim = self.cfg.gru_hidden_size * 2  # Bidirectional
        self.head = MultiHeadAttentionPooling(
            input_dim=gru_out_dim,
            num_heads=self.cfg.attention_heads,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
        )

    def _set_stride_to_one(self, block):
        """
        Recursively sets stride=(2,2) to (1,1) in Conv2d layers of a block.
        """
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
            # Also handle downsample blocks which might be Sequential(Conv, BN)
            # We rely on recursion to find the Conv inside the Sequential

    def forward(self, x):
        # Input x: (B, 3, F, T)

        # Backbone forward pass
        # Returns list of feature maps: [L2_out, L3_out, L4_out]
        # Due to modified strides, all should have same spatial dims as L2 (Stride 8 relative to input)
        feats = self.backbone(x)
        f2, f3, f4 = feats[0], feats[1], feats[2]

        # Global Average Pooling over Frequency dimension (dim 2)
        # Shape becomes (B, C, T)
        p2 = f2.mean(dim=2)
        p3 = f3.mean(dim=2)
        p4 = f4.mean(dim=2)

        # Project channels to common dimension
        p2 = self.proj_l2(p2)
        p3 = self.proj_l3(p3)
        p4 = self.proj_l4(p4)

        # Hierarchical Fusion
        # Concatenate along channel dimension: (B, 3*proj_dim, T)
        fused = torch.cat([p2, p3, p4], dim=1)

        # Permute for GRU: (B, T, C)
        fused = fused.permute(0, 2, 1)

        # BiGRU
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(fused)

        # Multi-Head Attention Pooling & Classification
        logits = self.head(gru_out)

        return logits
