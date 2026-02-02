import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Inception Style).
    Extracts features using parallel kernels of sizes 3, 5, and 7.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bottleneck to reduce computational cost
        bottleneck_channels = out_channels // 4
        self.use_bottleneck = in_channels > bottleneck_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
            if self.use_bottleneck
            else nn.Identity()
        )

        c_in = bottleneck_channels if self.use_bottleneck else in_channels

        # Parallel Convolutional Branches
        self.conv3 = nn.Conv1d(c_in, out_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(c_in, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(c_in, out_channels // 4, kernel_size=7, padding=3)

        # Pooling Branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(c_in, out_channels // 4, kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x_in = self.bottleneck(x)

        x3 = self.conv3(x_in)
        x5 = self.conv5(x_in)
        x7 = self.conv7(x_in)

        x_pool = self.maxpool(x_in)
        x_pool = self.conv_pool(x_pool)

        # Concatenate along channel dimension
        out = torch.cat([x3, x5, x7, x_pool], dim=1)
        out = self.bn(out)
        out = self.act(out)
        return out


class EEGEncoder(nn.Module):
    """
    Raw EEG Waveform Encoder.
    Input: (Batch, 20, Time)
    Output: Global Feature Vector
    """

    def __init__(self, in_channels=20, base_filters=64):
        super().__init__()
        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Inception Stages
        self.block1 = InceptionBlock1D(base_filters, base_filters * 2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.block2 = InceptionBlock1D(base_filters * 2, base_filters * 4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.block3 = InceptionBlock1D(base_filters * 4, base_filters * 8)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.out_dim = base_filters * 8  # e.g., 64 * 8 = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)

        # Flatten: (B, C, 1) -> (B, C)
        feat = self.global_pool(x).flatten(1)
        return feat


class CrossAttention(nn.Module):
    """
    Asymmetric Cross Attention Module.
    Query: EEG Features
    Key/Value: Spectrogram Spatial Features
    """

    def __init__(self, query_dim, key_dim, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        # query: (B, Q_dim) -> We treat this as sequence length 1
        # key_value: (B, Seq_len, K_dim)

        B = query.shape[0]

        # Project Q, K, V
        q = self.q_proj(query).unsqueeze(1)  # (B, 1, E)
        k = self.k_proj(key_value)  # (B, S, E)
        v = self.v_proj(key_value)  # (B, S, E)

        # Reshape for multi-head attention
        q = q.reshape(B, 1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, H, 1, D)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, H, S, D)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, H, S, D)

        # Scaled Dot-Product Attention
        scale = self.head_dim**-0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, 1, S)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = attn_weights @ v  # (B, H, 1, D)

        # Combine heads
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            B, 1, self.embed_dim
        )  # (B, 1, E)
        attn_output = self.out_proj(attn_output).squeeze(1)  # (B, E)

        # Residual connection + Norm
        # Note: We add the projected query as the residual base
        return self.norm(attn_output + self.q_proj(query))


class DeepSupervisedModel(nn.Module):
    """
    Deeply-Supervised Coordinate-Fusion Network.
    Integrates EEG and Spectrogram streams with Cross-Attention and Tri-Head supervision.
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        eeg_channels=Config.EEG_CHANNELS,
        spec_channels=Config.SPEC_CHANNELS,
    ):
        super().__init__()

        # --- Stream A: EEG ---
        self.eeg_encoder = EEGEncoder(in_channels=eeg_channels, base_filters=64)
        self.eeg_dim = self.eeg_encoder.out_dim  # 512

        # --- Stream B: Spectrogram ---
        # EfficientNet-B0 with 5 input channels (4 regions + 1 coordinate map)
        # global_pool='' ensures we get spatial feature maps for attention
        self.spec_encoder = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            in_chans=spec_channels,
            num_classes=0,
            global_pool="",
        )

        # Determine Spec feature dimension dynamically
        with torch.no_grad():
            dummy = torch.randn(1, spec_channels, 256, 256)
            feat = self.spec_encoder(dummy)
            self.spec_dim = feat.shape[1]  # Typically 1280 for B0

        self.spec_global_pool = nn.AdaptiveAvgPool2d(1)

        # --- Fusion: Cross Attention ---
        # EEG (Query) attends to Spec (Keys/Values)
        self.attn_dim = 256
        self.cross_attn = CrossAttention(
            query_dim=self.eeg_dim, key_dim=self.spec_dim, embed_dim=self.attn_dim
        )

        # --- Classification Heads ---
        # 1. EEG Auxiliary Head
        self.head_eeg = nn.Linear(self.eeg_dim, num_classes)

        # 2. Spec Auxiliary Head
        self.head_spec = nn.Linear(self.spec_dim, num_classes)

        # 3. Joint Head
        # Concatenates global features from both streams + attention context
        joint_input_dim = self.eeg_dim + self.spec_dim + self.attn_dim
        self.head_joint = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(joint_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_eeg, x_spec):
        # --- EEG Stream ---
        # x_eeg: (B, 20, 5000)
        eeg_feat = self.eeg_encoder(x_eeg)  # (B, 512)
        eeg_logit = self.head_eeg(eeg_feat)

        # --- Spec Stream ---
        # x_spec: (B, 5, 512, 512)
        spec_map = self.spec_encoder(x_spec)  # (B, 1280, H', W')

        # Global Pooling for Spec Head and Fusion
        spec_global = self.spec_global_pool(spec_map).flatten(1)  # (B, 1280)
        spec_logit = self.head_spec(spec_global)

        # --- Fusion ---
        # Flatten spatial dimensions for attention keys/values
        # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = spec_map.shape
        spec_tokens = spec_map.view(B, C, -1).permute(0, 2, 1)

        # Cross Attention: EEG queries Spec
        attn_context = self.cross_attn(eeg_feat, spec_tokens)  # (B, 256)

        # Concatenate all context vectors
        joint_feat = torch.cat([eeg_feat, spec_global, attn_context], dim=1)
        joint_logit = self.head_joint(joint_feat)

        # Return all logits for Deep Supervision Loss
        return joint_logit, eeg_logit, spec_logit
