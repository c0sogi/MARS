import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from library.config import Config


class ProjectionStem(nn.Module):
    """
    A simple projection stem that maps inputs to a hidden dimension using:
    Linear -> Permute -> Conv1d -> ReLU -> Dropout.
    Cite solution_lesson_node_00010: Correct ordering of operations.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.5):
        super(ProjectionStem, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        # Padding to maintain temporal dimension
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InChannels)
        x = self.linear(x)  # (B, T, OutChannels)
        x = x.permute(0, 2, 1)  # (B, OutChannels, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T, OutChannels)
        x = self.act(x)
        x = self.dropout(x)
        return x


class MultiStreamGRU(nn.Module):
    """
    Multi-Stream GRU Model.
    Cite solution_lesson_node_00029: Simpler architecture outperforms complex ones on small data.
    Cite solution_lesson_node_00021: Decoupled input stems.

    Architecture:
    1. Decoupled Projection Stems (Skeleton & Audio).
    2. Feature Fusion with Layer Normalization.
    3. Bidirectional GRU Backbone.
    4. Frame-wise Linear Classifier.
    """

    def __init__(self):
        super(MultiStreamGRU, self).__init__()

        # --- 1. Decoupled Input Stems ---
        # Skeleton Stream
        # Input: (B, T, 60) -> Output: (B, T, 128)
        self.skel_stem = ProjectionStem(
            in_channels=Config.SKELETON_INPUT_SIZE,
            out_channels=128,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # Audio Stream
        # Input: (B, T, 13) -> Output: (B, T, 64)
        self.audio_stem = ProjectionStem(
            in_channels=Config.AUDIO_INPUT_SIZE,
            out_channels=64,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # --- 2. Fusion ---
        # Concatenation of stem outputs
        fusion_dim = 128 + 64
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # --- 3. Recurrent Core ---
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_GRU_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.NUM_GRU_LAYERS > 1 else 0,
        )

        # --- 4. Classifier ---
        gru_out_dim = (
            Config.HIDDEN_SIZE * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_SIZE
        )
        self.classifier = nn.Linear(gru_out_dim, Config.NUM_CLASSES)

    def forward(self, skeleton, audio, lengths=None):
        """
        Forward pass of the model.

        Args:
            skeleton (torch.Tensor): Shape (Batch, Time, 60)
            audio (torch.Tensor): Shape (Batch, Time, 13)
            lengths (torch.Tensor, optional): Shape (Batch,). Valid lengths for packing.

        Returns:
            logits (torch.Tensor): Shape (Batch, Time, NumClasses)
        """
        # 1. Feature Extraction (Decoupled Stems)
        skel_features = self.skel_stem(skeleton)  # (B, T, 128)
        audio_features = self.audio_stem(audio)  # (B, T, 64)

        # 2. Fusion
        fused = torch.cat([skel_features, audio_features], dim=2)  # (B, T, 192)
        fused = self.fusion_norm(fused)

        # 3. Recurrent Processing
        if lengths is not None:
            # Pack sequence for efficient RNN processing ignoring padding
            # Ensure lengths are on CPU and int64 for pack_padded_sequence
            lengths_cpu = lengths.cpu().to(torch.int64)
            packed_input = rnn_utils.pack_padded_sequence(
                fused, lengths_cpu, batch_first=True, enforce_sorted=False
            )

            packed_output, _ = self.gru(packed_input)

            # Unpack back to padded sequence
            gru_out, _ = rnn_utils.pad_packed_sequence(
                packed_output, batch_first=True, total_length=fused.size(1)
            )
        else:
            gru_out, _ = self.gru(fused)

        # 4. Classification
        logits = self.classifier(gru_out)  # (B, T, NumClasses)

        return logits
