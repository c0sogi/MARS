import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library import config


class TemporalStream(nn.Module):
    """
    Independent temporal processing stream for a single modality.
    Structure: Linear -> Conv1d -> ReLU -> Dropout -> BiGRU.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(TemporalStream, self).__init__()

        # Projection to hidden dimension
        self.project = nn.Linear(input_dim, hidden_dim)

        # Temporal Convolution (Mid-Fusion preparation)
        # Padding ensures output length equals input length
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Bidirectional GRU for dynamics
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, lengths):
        """
        Args:
            x: (Batch, Time, InputDim)
            lengths: (Batch,) Tensor containing sequence lengths
        Returns:
            (Batch, Time, 2 * HiddenDim)
        """
        B, T, _ = x.size()

        # 1. Projection
        x = self.project(x)  # (B, T, H)

        # 2. Conv1d (Requires B, C, T)
        x = x.permute(0, 2, 1)  # (B, H, T)
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (B, T, H)

        # 3. BiGRU with Packing
        # Enforce lengths to CPU for compatibility with pack_padded_sequence
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.gru(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)

        return output


class ContextGating(nn.Module):
    """
    Context Gating mechanism: Y = X * Sigmoid(W*X + b)
    Dynamically re-weights features based on context.
    """

    def __init__(self, input_dim):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sigmoid(self.fc(x))
        return x * gates


class IDGFN(nn.Module):
    """
    Independent-Dynamics Gated Fusion Network.
    """

    def __init__(self):
        super(IDGFN, self).__init__()

        # --- 1. Decoupled Temporal Stems ---
        # Skeleton Stream
        self.skel_stream = TemporalStream(
            input_dim=60,  # 20 joints * 3 coordinates
            hidden_dim=config.HIDDEN_DIM_STREAM,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT_RATE,
        )

        # Audio Stream
        self.audio_stream = TemporalStream(
            input_dim=config.N_MFCC,
            hidden_dim=config.HIDDEN_DIM_STREAM,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT_RATE,
        )

        # Calculate Fusion Dimension
        # Each stream outputs 2 * HIDDEN_DIM_STREAM (Bidirectional)
        # We concatenate 2 streams
        self.fusion_dim = 2 * config.HIDDEN_DIM_STREAM * 2  # 128 * 2 * 2 = 512

        # --- 2. Gated Fusion ---
        self.ln = nn.LayerNorm(self.fusion_dim)
        self.cg = ContextGating(self.fusion_dim)

        # --- 3. Joint Integration Backbone ---
        # Residual BiGRU
        # Input: 512, Hidden: 256, Output (Bi): 512
        # Dimensions match for residual addition
        self.backbone = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=config.HIDDEN_DIM_BACKBONE,
            batch_first=True,
            bidirectional=True,
        )

        # --- 4. Non-Linear Classification Head ---
        # Input: 512 -> Output: NUM_CLASSES
        self.head = nn.Sequential(
            nn.Linear(2 * config.HIDDEN_DIM_BACKBONE, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (Batch, Time, 60)
            audio: (Batch, Time, 13)
            lengths: (Batch,)
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # 1. Independent Processing
        skel_feat = self.skel_stream(skeleton, lengths)  # (B, T, 256)
        audio_feat = self.audio_stream(audio, lengths)  # (B, T, 256)

        # 2. Late Temporal Fusion
        fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, 512)
        fused = self.ln(fused)
        fused = self.cg(fused)

        # 3. Backbone with Residual Connection
        packed_fused = pack_padded_sequence(
            fused, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_backbone, _ = self.backbone(packed_fused)

        backbone_out, _ = pad_packed_sequence(
            packed_backbone, batch_first=True, total_length=fused.size(1)
        )

        # Residual Add: Input (512) + Output (512)
        residual_out = fused + backbone_out

        # 4. Classification
        logits = self.head(residual_out)

        return logits
