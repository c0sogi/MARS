import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class Inception1DBlock(nn.Module):
    """
    A single Inception-style block for 1D data.
    Consists of parallel convolution branches with different kernel sizes,
    followed by concatenation, batch norm, activation, and pooling.
    """

    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()

        # Calculate channels per branch
        ch_per_branch = out_channels // len(kernels)
        self.out_channels = ch_per_branch * len(kernels)

        for k in kernels:
            # Padding = k // 2 maintains temporal dimension for odd kernels with stride 1
            self.branches.append(
                nn.Conv1d(
                    in_channels, ch_per_branch, kernel_size=k, padding=k // 2, stride=1
                )
            )

        self.bn = nn.BatchNorm1d(self.out_channels)
        self.act = nn.SiLU()  # Swish activation
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class Inception1DEncoder(nn.Module):
    """
    Multi-Scale 1D Convolutional Encoder for raw EEG data.
    Downsamples the temporal dimension while increasing channel depth.
    """

    def __init__(self, input_channels=20, embed_dim=512):
        super().__init__()

        # Sequence of Inception blocks
        # Input: (B, 20, 5000)
        self.blocks = nn.Sequential(
            Inception1DBlock(input_channels, 64),  # -> (B, 64, 2500)
            Inception1DBlock(64, 128),  # -> (B, 128, 1250)
            Inception1DBlock(128, 256),  # -> (B, 256, 625)
            Inception1DBlock(256, 512),  # -> (B, 512, 312)
            Inception1DBlock(512, 512),  # -> (B, 512, 156)
        )

        # Final projection to embedding dimension
        self.proj = nn.Conv1d(512, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 20, 5000)
        x = self.blocks(x)
        x = self.proj(x)

        # Permute for Attention: (B, Channels, Time) -> (B, Time, Channels)
        x = x.permute(0, 2, 1)
        return x


class CoordinateEfficientNet(nn.Module):
    """
    EfficientNet-B0 backbone modified for 5-channel input (Spectrograms + Coordinate Map).
    Preserves spatial/frequency structure by flattening feature maps instead of global pooling.
    """

    def __init__(self, in_chans=5, embed_dim=512):
        super().__init__()

        # Load pretrained EfficientNet-B0
        # in_chans=5 will adapt the first conv layer
        # global_pool='' removes the final pooling layer
        # num_classes=0 removes the classifier head
        self.backbone = timm.create_model(
            Config.SPEC_BACKBONE,
            pretrained=Config.SPEC_PRETRAINED,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )

        # EfficientNet-B0 outputs 1280 channels at the final stage
        self.backbone_dim = 1280
        self.proj = nn.Linear(self.backbone_dim, embed_dim)

    def forward(self, x):
        # x: (B, 5, 512, 512)
        x = self.backbone(x)  # -> (B, 1280, 16, 16) for 512x512 input

        B, C, H, W = x.shape

        # Flatten spatial dimensions (Time x Frequency)
        # (B, C, H, W) -> (B, C, H*W)
        x = x.reshape(B, C, H * W)

        # Permute to (B, Sequence, Channels)
        x = x.permute(0, 2, 1)  # -> (B, 256, 1280)

        # Project to common embedding dimension
        x = self.proj(x)  # -> (B, 256, Embed_Dim)

        return x


class CrossAttentionFusion(nn.Module):
    """
    Asymmetric Cross-Attention Fusion Block.
    EEG acts as Query, Spectrogram acts as Key/Value.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, q, kv):
        # q: EEG features (B, L_eeg, D)
        # kv: Spectrogram features (B, L_spec, D)

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        # Attention: Query=EEG, Key=Spec, Value=Spec
        attn_out, _ = self.mha(query=q_norm, key=kv_norm, value=kv_norm)

        # Residual connection + Dropout
        q = q + self.dropout(attn_out)
        q = self.norm_out(q)

        # FFN + Residual
        ffn_out = self.ffn(q)
        q = q + self.dropout(ffn_out)
        q = self.norm_ffn(q)

        return q


class CyclicFusionNet(nn.Module):
    """
    Main Model Architecture: Cyclic-Subset Coordinate-Guided Fusion Network.
    Combines Raw EEG and Spectrograms via Cross-Attention.
    """

    def __init__(self):
        super().__init__()

        # Stream A: EEG
        self.eeg_encoder = Inception1DEncoder(
            input_channels=Config.EEG_CHANNELS, embed_dim=Config.EMBED_DIM
        )

        # Stream B: Spectrogram
        self.spec_encoder = CoordinateEfficientNet(
            in_chans=Config.SPEC_CHANNELS, embed_dim=Config.EMBED_DIM
        )

        # Fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=Config.EMBED_DIM,
            num_heads=Config.ATTENTION_HEADS,
            dropout=Config.DROPOUT,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.EMBED_DIM // 2, Config.NUM_CLASSES),
        )

    def forward(self, eeg, spec):
        # eeg: (B, 20, 5000)
        # spec: (B, 5, 512, 512)

        # 1. Feature Extraction
        eeg_feats = self.eeg_encoder(eeg)  # (B, 156, 512)
        spec_feats = self.spec_encoder(spec)  # (B, 256, 512)

        # 2. Modality Dropout (Training Only)
        if self.training and Config.MODALITY_DROPOUT_PROB > 0:
            # Independent dropout for each stream
            r = torch.rand(1).item()
            if r < Config.MODALITY_DROPOUT_PROB:
                # Drop EEG (Query becomes zero -> Attention output becomes Mean(Value))
                eeg_feats = torch.zeros_like(eeg_feats)
            elif r < 2 * Config.MODALITY_DROPOUT_PROB:
                # Drop Spec (Key/Value becomes zero -> Attention output becomes Zero -> Residual preserves Query)
                spec_feats = torch.zeros_like(spec_feats)

        # 3. Fusion
        # EEG queries Spectrogram context
        fused_feats = self.fusion(q=eeg_feats, kv=spec_feats)  # (B, 156, 512)

        # 4. Global Average Pooling over time
        pooled = torch.mean(fused_feats, dim=1)  # (B, 512)

        # 5. Classification
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=1)

        return probs
