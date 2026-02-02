import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputStem(nn.Module):
    """
    Modality-specific processing stem.
    Structure: Linear Projection -> Temporal Conv1d(k=7) -> ReLU -> Dropout
    """

    def __init__(self, input_dim, embed_dim, kernel_size=7, dropout=0.3):
        super(InputStem, self).__init__()

        # Initial projection to embedding dimension
        self.project = nn.Linear(input_dim, embed_dim)

        # Temporal Convolution
        # padding='same' ensures output length matches input length
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Linear Projection
        x = self.project(x)  # (Batch, Time, EmbedDim)

        # Transpose for Conv1d: (Batch, EmbedDim, Time)
        x = x.transpose(1, 2)

        # Convolution
        x = self.conv(x)

        # Transpose back: (Batch, Time, EmbedDim)
        x = x.transpose(1, 2)

        # Activation and Dropout
        x = self.relu(x)
        x = self.dropout(x)

        return x


class ContextGating(nn.Module):
    """
    Context Gating mechanism: Y = X * sigmoid(W X + b)
    Acts as a 'Gate-Once' noise filter.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.gate = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (Batch, Time, Dim)
        gates = torch.sigmoid(self.gate(x))
        return x * gates


class SCRNet(nn.Module):
    """
    Skip-Context Residual Network (SCR-Net).

    Architecture:
    1. Decoupled Input Stems (Skeleton & Audio)
    2. Gated Fusion (Concat -> LayerNorm -> ContextGating)
    3. Skip-Context Backbone (BiGRU -> Skip-Context Injection -> BiGRU)
    4. Non-Linear Output Head
    """

    def __init__(self):
        super(SCRNet, self).__init__()

        # 1. Input Stems
        self.skel_stem = InputStem(
            input_dim=Config.SKELETON_INPUT_DIM,
            embed_dim=Config.SKELETON_EMBED_DIM,
            dropout=Config.DROPOUT,
        )

        self.audio_stem = InputStem(
            input_dim=Config.AUDIO_N_MFCC,
            embed_dim=Config.AUDIO_EMBED_DIM,
            dropout=Config.DROPOUT,
        )

        # 2. Gated Fusion
        self.fusion_dim = Config.FUSION_DIM  # 64 + 64 = 128
        self.ln = nn.LayerNorm(self.fusion_dim)
        self.context_gating = ContextGating(self.fusion_dim)

        # 3. Skip-Context Backbone
        self.hidden_dim = Config.HIDDEN_DIM  # 256
        self.gru_output_dim = self.hidden_dim * 2  # Bidirectional -> 512

        # Layer 1: BiGRU
        self.gru1 = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Skip-Context Projection: Projects Fusion (128) to GRU Output space (512)
        self.skip_project = nn.Linear(self.fusion_dim, self.gru_output_dim)

        # Layer 2: BiGRU
        # Input size is 512 because it receives (H1 + Projected_Fusion)
        self.gru2 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        # Input: GRU2 output (512)
        self.head = nn.Sequential(
            nn.Linear(self.gru_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, Config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths=None):
        # skeleton: (Batch, Time, SkelDim)
        # audio: (Batch, Time, AudioDim)

        # 1. Process Stems
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        # 2. Fusion
        # Concatenate
        fused = torch.cat([skel_feat, audio_feat], dim=-1)  # (Batch, Time, 128)

        # Normalize and Gate
        fused = self.ln(fused)
        fused = self.context_gating(fused)  # This is 'F'

        # 3. Backbone Layer 1
        # We use packed sequences if lengths are provided for efficiency in RNNs,
        # but for the residual connection math (H1 + Proj(F)), unpacked is easier.
        # Given the task constraints and batch size, unpacked processing is acceptable.
        # However, using pack_padded_sequence is better for correctness with padding.

        if lengths is not None:
            # Move lengths to CPU for packing
            lens_cpu = lengths.cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                fused, lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_h1, _ = self.gru1(packed_input)
            h1, _ = nn.utils.rnn.pad_packed_sequence(packed_h1, batch_first=True)
        else:
            h1, _ = self.gru1(fused)

        # h1: (Batch, Time, 512)

        # 4. Skip-Context Injection
        # Input_2 = H1 + Linear(F)
        # We need to ensure F matches H1's length if padding was stripped/restored,
        # but here F and H1 are aligned in time.

        f_proj = self.skip_project(fused)  # (Batch, Time, 512)
        input_2 = h1 + f_proj

        # 5. Backbone Layer 2
        if lengths is not None:
            packed_input_2 = nn.utils.rnn.pack_padded_sequence(
                input_2, lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_h2, _ = self.gru2(packed_input_2)
            h2, _ = nn.utils.rnn.pad_packed_sequence(packed_h2, batch_first=True)
        else:
            h2, _ = self.gru2(input_2)

        # h2: (Batch, Time, 512)

        # 6. Output Head
        logits = self.head(h2)  # (Batch, Time, NumClasses)

        return logits
