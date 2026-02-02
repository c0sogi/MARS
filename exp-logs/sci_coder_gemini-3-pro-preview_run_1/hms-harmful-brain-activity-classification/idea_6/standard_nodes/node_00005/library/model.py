import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    Inception-style 1D Convolutional Block.
    Applies parallel convolutions with different kernel sizes to capture multi-scale features.
    """

    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7], dropout=0.0):
        super().__init__()

        # Ensure out_channels is divisible by the number of branches (kernels + 1 pooling branch)
        num_branches = len(kernels) + 1
        branch_channels = out_channels // num_branches

        self.branches = nn.ModuleList()

        # Convolutional branches
        for k in kernels:
            # Padding to maintain temporal dimension: (k - 1) // 2
            pad = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels, branch_channels, kernel_size=k, padding=pad, bias=False
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(),
            )
            self.branches.append(branch)

        # Pooling branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Handle remainder channels if out_channels not perfectly divisible
        self.final_channels = branch_channels * num_branches

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        out = torch.cat(outputs, dim=1)
        return self.dropout(out)


class Siamese1DEncoder(nn.Module):
    """
    Shared-weight encoder for processing Left and Right EEG groups.
    """

    def __init__(
        self, in_channels, base_filters=Config.CNN_FILTERS, kernels=Config.CNN_KERNELS
    ):
        super().__init__()

        # Layer 1
        self.block1 = InceptionBlock1D(in_channels, base_filters, kernels)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        # Layer 2
        in_ch2 = self.block1.final_channels
        self.block2 = InceptionBlock1D(in_ch2, base_filters * 2, kernels)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        # Layer 3
        in_ch3 = self.block2.final_channels
        self.block3 = InceptionBlock1D(in_ch3, base_filters * 4, kernels)
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        # Layer 4
        in_ch4 = self.block3.final_channels
        self.block4 = InceptionBlock1D(in_ch4, base_filters * 4, kernels)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.out_channels = self.block4.final_channels

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return x


class SpectrogramEncoder(nn.Module):
    """
    2D CNN Encoder for Spectrograms using EfficientNet backbone.
    """

    def __init__(
        self, model_name=Config.SPEC_BACKBONE, pretrained=Config.SPEC_PRETRAINED
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(4,),  # Get features from the last stage
        )

        # Determine output channels dynamically
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # features is a list of tensors, we took the last one
            self.out_channels = features[0].shape[1]

    def forward(self, x):
        # x: (Batch, 3, H, W)
        features = self.backbone(x)[0]  # (Batch, C, H', W')
        return features


class CrossAttentionFusion(nn.Module):
    """
    Cross Attention Module.
    Query: EEG Features
    Key/Value: Spectrogram Features
    """

    def __init__(self, query_dim, key_dim, embed_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_val):
        # query: (Batch, Time, Q_Dim)
        # key_val: (Batch, Pixels, K_Dim)

        q = self.query_proj(query)
        k = self.key_proj(key_val)
        v = self.value_proj(key_val)

        # Attention(Q, K, V)
        attn_out, _ = self.attn(q, k, v)

        # Add & Norm
        x = self.norm(q + attn_out)

        # FFN
        x2 = self.ffn(x)
        x = self.norm2(x + x2)

        return x


class SymmetryAwareNet(nn.Module):
    """
    Symmetry-Aware Siamese Dual-Stream Network.
    Combines Siamese EEG processing with Global Spectrogram Context.
    """

    def __init__(self):
        super().__init__()

        # 1. Siamese EEG Encoder
        # Input channels = number of electrodes in one hemisphere group
        eeg_in_channels = len(Config.LEFT_HEMISPHERE_CHANNELS)
        self.eeg_encoder = Siamese1DEncoder(eeg_in_channels)

        # 2. Spectrogram Encoder
        self.spec_encoder = SpectrogramEncoder()

        # 3. Dimensionality Setup
        # EEG features come out as (B, C, T).
        # We concat Sum and Diff features, so channels double.
        self.eeg_feat_dim = self.eeg_encoder.out_channels * 2
        self.spec_feat_dim = self.spec_encoder.out_channels
        self.fusion_dim = Config.ATTENTION_DIM

        # 4. Fusion
        self.fusion = CrossAttentionFusion(
            query_dim=self.eeg_feat_dim,
            key_dim=self.spec_feat_dim,
            embed_dim=self.fusion_dim,
        )

        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, Config.NUM_CLASSES),
        )

    def forward(self, left_eeg, right_eeg, spectrogram):
        # --- Stream A: Siamese EEG ---
        # left_eeg, right_eeg: (B, C_in, T_in)

        feat_left = self.eeg_encoder(left_eeg)  # (B, C_enc, T_out)
        feat_right = self.eeg_encoder(right_eeg)  # (B, C_enc, T_out)

        # Lateralization Module
        # Generalized Features (Sum)
        feat_gen = feat_left + feat_right
        # Lateralized Features (Difference)
        feat_lat = feat_left - feat_right

        # Concatenate: (B, 2*C_enc, T_out)
        eeg_features = torch.cat([feat_gen, feat_lat], dim=1)

        # Permute for Attention: (B, T_out, 2*C_enc)
        eeg_features = eeg_features.permute(0, 2, 1)

        # --- Stream B: Spectrogram ---
        # spectrogram: (B, 3, H, W)
        spec_features = self.spec_encoder(spectrogram)  # (B, C_spec, H', W')

        # Flatten spatial dimensions for attention Key/Value
        B, C, H, W = spec_features.shape
        spec_features = spec_features.view(B, C, -1).permute(
            0, 2, 1
        )  # (B, H*W, C_spec)

        # --- Fusion ---
        # Query = EEG (Time), Key/Value = Spec (Space)
        fused_features = self.fusion(
            eeg_features, spec_features
        )  # (B, T_out, Fusion_Dim)

        # --- Classification ---
        # Global Average Pooling over the temporal dimension
        pooled = torch.mean(fused_features, dim=1)  # (B, Fusion_Dim)

        logits = self.classifier(pooled)

        return logits
