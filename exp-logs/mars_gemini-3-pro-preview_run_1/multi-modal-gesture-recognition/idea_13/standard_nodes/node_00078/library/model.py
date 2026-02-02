import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputStem(nn.Module):
    """
    Independent processing stem for a single modality.
    Structure: Linear -> Permute -> Conv1d(k=7) -> ReLU -> Dropout -> Permute
    """

    def __init__(self, input_dim, output_dim, kernel_size=7, dropout=0.3):
        super(InputStem, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

        # Calculate padding to keep sequence length unchanged: (k - 1) / 2
        # Assuming stride=1 and dilation=1
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Time, Input_Dim)

        # 1. Linear Projection
        x = self.projection(x)  # (Batch, Time, Output_Dim)

        # 2. Permute for Conv1d (Batch, Channels, Time)
        x = x.permute(0, 2, 1)

        # 3. Temporal Convolution
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 4. Permute back (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        return x


class ContextGating(nn.Module):
    """
    Context Gating mechanism: Y = X * Sigmoid(W*X + b)
    Filters noise/unreliable features based on context.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x shape: (Batch, Time, Dimension)
        gate = torch.sigmoid(self.fc(x))
        return x * gate


class IICGRN(nn.Module):
    """
    Input-Injected Context-Gated Recurrent Network.

    Features:
    1. Decoupled Input Stems
    2. Gated Fusion
    3. Input-Injected BiGRU Backbone
    4. Non-Linear Output Head
    """

    def __init__(self):
        super(IICGRN, self).__init__()

        # Dimensions from Config
        self.skel_in = Config.SKELETON_INPUT_SIZE
        self.audio_in = Config.N_MFCC
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_classes = Config.NUM_CLASSES
        self.dropout_p = Config.DROPOUT

        # Define stem dimension (split hidden size roughly equally)
        # We use HIDDEN_SIZE // 2 for each stem so concatenation results in HIDDEN_SIZE
        stem_dim = self.hidden_size // 2

        # 1. Input Stems
        self.skel_stem = InputStem(
            self.skel_in, stem_dim, kernel_size=7, dropout=self.dropout_p
        )
        self.audio_stem = InputStem(
            self.audio_in, stem_dim, kernel_size=7, dropout=self.dropout_p
        )

        # Fusion dimension
        fusion_dim = stem_dim * 2

        # 2. Fusion & Gating
        self.ln = nn.LayerNorm(fusion_dim)
        self.context_gating = ContextGating(fusion_dim)

        # 3. Input-Injected Backbone
        # We use separate GRU layers to allow manual injection of input into the second layer.

        # Layer 1: Processes Fused Features (F)
        # Bidirectional output size = hidden_size * 2
        self.gru1 = nn.GRU(
            input_size=fusion_dim,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        gru_out_dim = self.hidden_size * 2

        # Injection Projection: Projects F (fusion_dim) to match GRU output size (gru_out_dim)
        self.injection_proj = nn.Linear(fusion_dim, gru_out_dim)

        # Layer 2: Processes (H1 + Proj(F))
        self.gru2 = nn.GRU(
            input_size=gru_out_dim,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        # 4. Output Head
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, skeleton, audio, lengths=None):
        """
        Args:
            skeleton: (Batch, Time, 60)
            audio: (Batch, Time, 13)
            lengths: (Batch,) - Optional, not strictly used due to 'no-masking' strategy
        """

        # --- 1. Independent Processing ---
        s_feat = self.skel_stem(skeleton)
        a_feat = self.audio_stem(audio)

        # --- 2. Fusion ---
        # Concatenate along feature dimension
        fused = torch.cat([s_feat, a_feat], dim=-1)  # (Batch, Time, FusionDim)

        # Normalize
        fused = self.ln(fused)

        # Context Gating (This is 'F')
        F_tensor = self.context_gating(fused)

        # --- 3. Input-Injected Recurrence ---

        # Layer 1
        # We process the full padded sequence as per "do not mask padded time-steps" requirement
        H1, _ = self.gru1(F_tensor)  # H1: (Batch, Time, 2*Hidden)

        # Input Injection
        # Project original features F to match H1 dimensions
        F_proj = self.injection_proj(F_tensor)

        # Add projection to H1 (Residual-like injection)
        H2_in = H1 + F_proj

        # Layer 2
        H2, _ = self.gru2(H2_in)  # H2: (Batch, Time, 2*Hidden)

        # --- 4. Output Head ---
        logits = self.head(H2)  # (Batch, Time, NumClasses)

        return logits
