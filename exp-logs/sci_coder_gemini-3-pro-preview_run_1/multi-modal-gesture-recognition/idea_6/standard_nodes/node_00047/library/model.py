import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ContextGating(nn.Module):
    """
    Context Gating block: Y = X * Sigmoid(Linear(X))
    Dynamically re-weights features based on their own values.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (B, T, D)
        gates = torch.sigmoid(self.fc(x))
        return x * gates


class InputStem(nn.Module):
    """
    Modality-specific processing stem.
    Linear -> Conv1d (k=7) -> ReLU -> Dropout
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(InputStem, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        # Padding = (kernel_size - 1) // 2 to maintain temporal length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)
        x = self.project(x)  # (B, T, HiddenDim)

        # Conv1d expects (B, Channels, Time)
        x = x.transpose(1, 2)  # (B, HiddenDim, T)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (B, T, HiddenDim)
        return x


class ACGRNet(nn.Module):
    """
    Attention-Augmented Context-Gated Residual Network.

    Structure:
    1. Independent Skeleton & Audio Stems
    2. Fusion (Concat + LN + ContextGating)
    3. Hybrid Backbone:
       - BiGRU (Local)
       - MHSA (Global)
       - BiGRU (Integration)
       - Residual (Stage 1 -> Stage 3)
    4. Classifier
    """

    def __init__(self):
        super(ACGRNet, self).__init__()

        # Hyperparameters
        skel_in = Config.INPUT_DIM_SKELETON
        audio_in = Config.INPUT_DIM_AUDIO
        hidden_dim = Config.HIDDEN_DIM
        kernel_size = Config.KERNEL_SIZE
        dropout = Config.DROPOUT
        num_heads = Config.NUM_HEADS
        num_classes = Config.NUM_CLASSES

        # 1. Input Stems
        self.skel_stem = InputStem(skel_in, hidden_dim, kernel_size, dropout)
        self.audio_stem = InputStem(audio_in, hidden_dim, kernel_size, dropout)

        # 2. Fusion
        # Concatenated dimension = hidden_dim * 2
        fusion_dim = hidden_dim * 2
        self.ln_fusion = nn.LayerNorm(fusion_dim)
        self.cg_fusion = ContextGating(fusion_dim)

        # 3. Hybrid Recurrent-Attention Backbone

        # Stage 1: Local Dynamics (BiGRU)
        # Input: fusion_dim, Output: hidden_dim * 2 (bidirectional)
        # We set hidden_size of GRU to fusion_dim // 2 so output matches fusion_dim
        gru_hidden = fusion_dim // 2
        self.gru_stage1 = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            bidirectional=True,
            batch_first=True,
        )

        # Stage 2: Global Context (MHSA)
        # Input/Output: fusion_dim
        self.mhsa = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_attn = nn.LayerNorm(fusion_dim)  # Pre-norm or Post-norm usually helps

        # Stage 3: Integration (BiGRU)
        self.gru_stage3 = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            bidirectional=True,
            batch_first=True,
        )

        # 4. Classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, skeleton, audio, mask=None):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 13)
            mask: (B, T) Boolean mask where True indicates valid frames, False indicates padding.
        """
        # 1. Process Stems
        skel_feat = self.skel_stem(skeleton)  # (B, T, H)
        audio_feat = self.audio_stem(audio)  # (B, T, H)

        # 2. Fusion
        fused = torch.cat([skel_feat, audio_feat], dim=-1)  # (B, T, 2H)
        fused = self.ln_fusion(fused)
        fused = self.cg_fusion(fused)

        # 3. Backbone

        # Stage 1: BiGRU
        # Pack padded sequence could be used here for efficiency, but for simplicity/compatibility
        # with the custom mask logic in MHSA, we stick to standard tensor ops or use mask implicitly.
        # PyTorch GRU handles padding fine if we don't pack, just output at padding positions is garbage but masked later.
        h1, _ = self.gru_stage1(fused)  # (B, T, 2H)

        # Stage 2: MHSA
        # key_padding_mask: True for values to be ignored.
        # Our `mask` input has True for valid, False for padding.
        # So we pass ~mask (inverted).
        key_padding_mask = ~mask if mask is not None else None

        # Self-Attention
        # Residual connection around attention is standard, but here we follow the prompt's specific
        # "Sandwich" structure. The prompt defines the residual from Stage 1 to Stage 3.
        # It doesn't explicitly forbid standard residual in MHSA, but let's stick to the prompt structure.
        # "Stage 2... is inserted here... Stage 3... integrates... Residual... links Stage 1 directly to Stage 3."

        attn_out, _ = self.mhsa(h1, h1, h1, key_padding_mask=key_padding_mask)
        # Apply LayerNorm after attention (standard practice)
        h2 = self.ln_attn(
            h1 + attn_out
        )  # Adding local residual here helps gradient flow through attention

        # Stage 3: BiGRU
        h3, _ = self.gru_stage3(h2)  # (B, T, 2H)

        # Residual Connection (Stage 1 -> Stage 3)
        # As per prompt: "Residual skip connection links the output of Stage 1 directly to the output of Stage 3"
        out_backbone = h3 + h1

        # 4. Classifier
        logits = self.classifier(out_backbone)  # (B, T, NumClasses)

        return logits
