import torch
import torch.nn as nn
import timm
from library.config import Config


class InceptionModule(nn.Module):
    """
    A 1D Inception Module with parallel convolutions of different kernel sizes.
    """

    def __init__(self, in_channels, out_channels, bottleneck_channels=32):
        super().__init__()
        # Branch 1: 1x1 Conv
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Branch 2: 1x1 -> 3x3 Conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=3, padding=1),
        )

        # Branch 3: 1x1 -> 5x5 Conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=5, padding=2),
        )

        # Branch 4: MaxPool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate along channel dimension
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.act(self.bn(out))


class EEGEncoder(nn.Module):
    """
    Multi-Scale 1D CNN for Raw EEG data.
    """

    def __init__(self, in_channels=20, base_filters=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
        )

        # Block 1
        self.block1 = InceptionModule(base_filters, base_filters)
        self.pool1 = nn.MaxPool1d(2)

        # Block 2
        in_c2 = base_filters * 4
        self.block2 = InceptionModule(in_c2, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)

        # Block 3
        in_c3 = (base_filters * 2) * 4
        self.block3 = InceptionModule(in_c3, base_filters * 4)
        self.pool3 = nn.MaxPool1d(2)

        # Final Dimensions
        self.out_dim = (base_filters * 4) * 4
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, Channels, Time)
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)

        # Sequence features for attention
        e_seq = x
        # Global features for aux head
        e_vec = self.global_pool(x).flatten(1)

        return e_seq, e_vec


class SpecEncoder(nn.Module):
    """
    EfficientNet Backbone for Spectrogram data.
    """

    def __init__(self, model_name="efficientnet_b0", in_channels=5, pretrained=True):
        super().__init__()
        # Load backbone, extract features only
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(4,),  # Get the last feature map
        )

        # Determine output channels dynamically
        self.out_dim = self.backbone.feature_info[-1]["num_chs"]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, 5, H, W)
        # Extract features
        feats = self.backbone(x)[0]  # (B, C, H', W')

        # Create tokens for attention: (B, N_tokens, C)
        b, c, h, w = feats.shape
        s_tok = feats.view(b, c, -1).permute(0, 2, 1)

        # Global vector
        s_vec = self.global_pool(feats).flatten(1)

        return s_tok, s_vec


class CrossAttentionFusion(nn.Module):
    """
    Unidirectional Cross Attention: EEG queries Spectrogram.
    """

    def __init__(self, eeg_dim, spec_dim, embed_dim=128, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(eeg_dim, embed_dim)
        self.key_proj = nn.Linear(spec_dim, embed_dim)
        self.value_proj = nn.Linear(spec_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, e_seq, s_tok):
        # e_seq: (B, C_eeg, T) -> Need (B, T, C_eeg) for projection
        # s_tok: (B, N, C_spec)

        q = self.query_proj(e_seq.permute(0, 2, 1))  # (B, T, embed)
        k = self.key_proj(s_tok)  # (B, N, embed)
        v = self.value_proj(s_tok)  # (B, N, embed)

        # Attention: Q queries K, retrieves V
        attn_out, _ = self.attn(q, k, v)  # (B, T, embed)

        # Pool over the temporal dimension to get a global context vector
        # (B, T, embed) -> (B, embed, T) -> Pool -> (B, embed)
        pooled = self.pool(attn_out.permute(0, 2, 1)).flatten(1)

        return self.norm(pooled)


class AuxiliaryFusionNet(nn.Module):
    """
    Main Architecture:
    - Stream A: EEG Encoder
    - Stream B: Spec Encoder
    - Fusion: Cross Attention + Concat
    - Outputs: Joint Head + 2 Aux Heads
    """

    def __init__(self):
        super().__init__()

        # 1. Encoders
        self.eeg_encoder = EEGEncoder(in_channels=Config.EEG_CHANNELS)
        self.spec_encoder = SpecEncoder(
            model_name=Config.BACKBONE_SPEC,
            in_channels=Config.SPEC_CHANNELS,
            pretrained=Config.PRETRAINED,
        )

        eeg_dim = self.eeg_encoder.out_dim
        spec_dim = self.spec_encoder.out_dim
        attn_dim = 128

        # 2. Fusion
        self.fusion = CrossAttentionFusion(eeg_dim, spec_dim, embed_dim=attn_dim)

        # 3. Heads
        # Joint Head Input: [Attention_Context, EEG_Global, Spec_Global]
        joint_in_dim = attn_dim + eeg_dim + spec_dim

        self.joint_head = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(joint_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, Config.NUM_CLASSES),
        )

        self.aux_eeg_head = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(eeg_dim, 128),
            nn.ReLU(),
            nn.Linear(128, Config.NUM_CLASSES),
        )

        self.aux_spec_head = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(spec_dim, 128),
            nn.ReLU(),
            nn.Linear(128, Config.NUM_CLASSES),
        )

    def forward(self, eeg, spec):
        # 1. Feature Extraction
        e_seq, e_vec = self.eeg_encoder(eeg)
        s_tok, s_vec = self.spec_encoder(spec)

        # 2. Modality Dropout (Training Only)
        # We create copies for the fusion layer to potentially zero out.
        # The aux heads always get the clean features to ensure gradient flow.
        e_seq_joint = e_seq
        s_tok_joint = s_tok
        e_vec_joint = e_vec
        s_vec_joint = s_vec

        if self.training:
            r = torch.rand(1).item()
            # Probability to drop EEG stream
            if r < Config.MODALITY_DROPOUT_PROB:
                e_seq_joint = torch.zeros_like(e_seq)
                e_vec_joint = torch.zeros_like(e_vec)
            # Probability to drop Spec stream (mutually exclusive or independent logic)
            # Here we use exclusive buckets for simplicity: [0, p) drop EEG, [p, 2p) drop Spec
            elif r < 2 * Config.MODALITY_DROPOUT_PROB:
                s_tok_joint = torch.zeros_like(s_tok)
                s_vec_joint = torch.zeros_like(s_vec)

        # 3. Fusion
        attn_out = self.fusion(e_seq_joint, s_tok_joint)

        # Concatenate Global Vectors
        joint_feat = torch.cat([attn_out, e_vec_joint, s_vec_joint], dim=1)

        # 4. Prediction
        joint_logits = self.joint_head(joint_feat)
        aux_eeg_logits = self.aux_eeg_head(e_vec)
        aux_spec_logits = self.aux_spec_head(s_vec)

        return joint_logits, aux_eeg_logits, aux_spec_logits
