import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StreamEncoder(nn.Module):
    """
    Processes a single modality stream (Pose, Velocity, or Audio).
    Structure: Linear -> Permute -> Conv1d -> ReLU -> Dropout -> Permute
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, padding, dropout):
        super(StreamEncoder, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.project(x)  # (Batch, Time, HiddenDim)

        # Permute for Conv1d: (Batch, HiddenDim, Time)
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Permute back: (Batch, Time, HiddenDim)
        x = x.permute(0, 2, 1)
        return x


class ContextGating(nn.Module):
    """
    Context Gating block: Y = X * Sigmoid(W*X + b)
    Dynamically suppresses unreliable features.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (Batch, Time, Dimension)
        gate = torch.sigmoid(self.fc(x))
        return x * gate


class ResidualBiGRU(nn.Module):
    """
    Residual Bidirectional GRU Backbone.
    Consists of 2 stacked BiGRU layers with a residual connection between their outputs.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(ResidualBiGRU, self).__init__()
        self.hidden_dim = hidden_dim

        # Layer 1
        self.gru1 = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Layer 2
        # Input to layer 2 is output of layer 1 (hidden_dim * 2)
        self.gru2 = nn.GRU(
            hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        # Layer 1
        out1, _ = self.gru1(x)
        out1 = self.dropout(out1)

        # Layer 2
        out2, _ = self.gru2(out1)
        out2 = self.dropout(out2)

        # Residual Connection
        # Both out1 and out2 have shape (Batch, Time, HiddenDim * 2)
        return out1 + out2


class CGRNet(nn.Module):
    """
    Context-Gated Residual Network (CGRNet).
    Dual-Stream Input (Pose, Audio) -> Gated Fusion -> Residual BiGRU -> Classification Head.
    Cite solution_lesson_node_00036
    """

    def __init__(self):
        super(CGRNet, self).__init__()

        # 1. Dual-Stream Encoders
        # Cite solution_lesson_node_00053: Removed Velocity Stream
        embed_dim = Config.HIDDEN_DIM

        self.pose_encoder = StreamEncoder(
            Config.POSE_INPUT_DIM,
            embed_dim,
            Config.CNN_KERNEL_SIZE,
            Config.CNN_PADDING,
            Config.DROPOUT,
        )

        self.audio_encoder = StreamEncoder(
            Config.AUDIO_INPUT_DIM,
            embed_dim,
            Config.CNN_KERNEL_SIZE,
            Config.CNN_PADDING,
            Config.DROPOUT,
        )

        # 2. Gated Fusion
        fused_dim = embed_dim * 2
        self.layer_norm = nn.LayerNorm(fused_dim)
        self.context_gating = ContextGating(fused_dim)

        # 3. Residual Recurrent Backbone
        self.backbone = ResidualBiGRU(
            input_dim=fused_dim, hidden_dim=Config.HIDDEN_DIM, dropout=Config.DROPOUT
        )

        # Backbone output dimension (Bidirectional)
        rnn_out_dim = Config.HIDDEN_DIM * 2

        # 4. Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_out_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(rnn_out_dim, Config.NUM_CLASSES),
        )

    def forward(self, pose, audio):
        """
        Args:
            pose: (B, T, D_pose)
            audio: (B, T, D_audio)
        Returns:
            class_logits: (B, T, NumClasses)
        """
        # 1. Encode Streams
        p_emb = self.pose_encoder(pose)  # (B, T, H)
        a_emb = self.audio_encoder(audio)  # (B, T, H)

        # 2. Fuse
        fused = torch.cat([p_emb, a_emb], dim=2)  # (B, T, 2H)
        fused = self.layer_norm(fused)
        fused = self.context_gating(fused)

        # 3. Backbone
        features = self.backbone(fused)  # (B, T, 2H)

        # 4. Heads
        class_logits = self.class_head(features)

        return class_logits
