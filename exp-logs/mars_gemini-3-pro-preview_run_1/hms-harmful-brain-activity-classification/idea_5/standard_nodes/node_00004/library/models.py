import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    Inception-style block for 1D EEG data.
    Applies parallel convolutions with different kernel sizes and a pooling branch.
    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes=[3, 5, 7], bottleneck_channels=32
    ):
        super().__init__()

        # Bottleneck to reduce computational cost
        self.use_bottleneck = in_channels > bottleneck_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
            if self.use_bottleneck
            else nn.Identity()
        )
        input_channels = bottleneck_channels if self.use_bottleneck else in_channels

        # Calculate filters per branch to approximate out_channels
        # Branches: len(kernel_sizes) convs + 1 pooling
        num_branches = len(kernel_sizes) + 1
        filters_per_branch = out_channels // num_branches

        # Convolutional branches
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        filters_per_branch,
                        kernel_size=k,
                        padding=k // 2,
                    ),
                    nn.BatchNorm1d(filters_per_branch),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )

        # Pooling branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, filters_per_branch, kernel_size=1),
            nn.BatchNorm1d(filters_per_branch),
            nn.GELU(),
        )

        # Final projection to ensure exact out_channels dimension
        current_channels = filters_per_branch * num_branches
        self.project = (
            nn.Conv1d(current_channels, out_channels, kernel_size=1)
            if current_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        x_bot = self.bottleneck(x)

        outputs = []
        for conv in self.convs:
            outputs.append(conv(x_bot))

        # Pooling branch operates on original input (standard Inception practice) or bottleneck
        # Here we operate on original to preserve features lost in bottleneck
        outputs.append(self.pool_branch(x))

        out = torch.cat(outputs, dim=1)
        out = self.project(out)
        return out


class MultiScale1DCNN(nn.Module):
    """
    Stream A: Micro-View Encoder.
    Processes raw EEG signals using stacked Inception blocks.
    """

    def __init__(self, in_channels=20, base_filters=64, embed_dim=256):
        super().__init__()

        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Encoder Blocks
        self.block1 = InceptionBlock1D(base_filters, base_filters * 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.block2 = InceptionBlock1D(base_filters * 2, base_filters * 4)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.block3 = InceptionBlock1D(base_filters * 4, base_filters * 8)

        # Global Pooling and Projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8, embed_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)

        x = self.global_pool(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class TriViewNet(nn.Module):
    """
    Tri-View Hierarchical Fusion Network.
    Integrates Raw EEG, Local Spectrogram, and Global Spectrogram.
    """

    def __init__(self, num_classes=6, embed_dim=256, pretrained=True):
        super().__init__()

        # --- Stream A: Micro (Raw EEG) ---
        self.micro_encoder = MultiScale1DCNN(
            in_channels=len(Config.EEG_CHANNELS), embed_dim=embed_dim
        )

        # --- Stream B: Meso (Local Spectrogram) ---
        # Extracts a global vector for the specific event
        self.meso_encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        self.meso_proj = nn.Sequential(
            nn.Linear(self.meso_encoder.num_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # --- Stream C: Macro (Global Spectrogram) ---
        # Extracts spatial/temporal feature maps (sequence) for context
        self.macro_encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # Keep spatial dimensions
            in_chans=3,
        )
        # Project channel dim to embed_dim
        self.macro_proj = nn.Conv2d(
            self.macro_encoder.num_features, embed_dim, kernel_size=1
        )

        # --- Hierarchical Fusion ---

        # 1. Event Representation (Micro + Meso)
        self.event_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # 2. Contextual Attention
        # Query: Event Vector (Batch, 1, Dim)
        # Key/Value: Macro Sequence (Batch, SeqLen, Dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # --- Classifier ---
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, micro, meso, macro):
        """
        Args:
            micro: (B, 20, 5000) Raw EEG
            meso: (B, 3, 224, 224) Local Spectrogram
            macro: (B, 3, 512, 512) Global Spectrogram
        """

        # 1. Encode Micro (EEG) -> (B, Embed)
        micro_emb = self.micro_encoder(micro)

        # 2. Encode Meso (Local Spec) -> (B, Embed)
        meso_feat = self.meso_encoder(meso)
        meso_emb = self.meso_proj(meso_feat)

        # 3. Encode Macro (Global Spec) -> (B, Embed, H, W)
        macro_feat = self.macro_encoder(macro)
        macro_feat = self.macro_proj(macro_feat)

        # Flatten Macro to Sequence: (B, Embed, H, W) -> (B, H*W, Embed)
        B, C, H, W = macro_feat.shape
        macro_seq = macro_feat.flatten(2).transpose(1, 2)

        # 4. Create Event Query
        # Concatenate Micro and Meso
        event_raw = torch.cat([micro_emb, meso_emb], dim=1)
        event_query = self.event_fusion(event_raw).unsqueeze(1)  # (B, 1, Embed)

        # 5. Cross Attention (Event queries Context)
        # Output: (B, 1, Embed)
        attn_out, _ = self.cross_attn(query=event_query, key=macro_seq, value=macro_seq)

        # Residual Connection + Norm
        fused_repr = self.attn_norm(event_query + attn_out).squeeze(1)  # (B, Embed)

        # 6. Classification
        logits = self.classifier(fused_repr)

        return logits
