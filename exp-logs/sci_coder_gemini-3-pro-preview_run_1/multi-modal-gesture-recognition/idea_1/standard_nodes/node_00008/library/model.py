import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_CLASSES,
    DROPOUT,
    BIDIRECTIONAL,
    CNN_KERNEL_SIZE,
)


class GestureGRU(nn.Module):
    """
    Frame-wise Supervised Gated Recurrent Unit (GRU) for Gesture Recognition.

    Architecture:
    1. Feature Fusion: Projects multimodal inputs (Skeleton + Audio) to a hidden embedding.
    2. Recurrent Core: GRU layers to capture temporal dependencies.
    3. Classification Head: Projects temporal states to class logits for each frame.
    """

    def __init__(self):
        super(GestureGRU, self).__init__()

        # 1. Feature Fusion Layer
        # Projects concatenated input features to the hidden dimension
        self.feature_fusion = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT)
        )

        # 1.5 Temporal Convolution (Cite solution_lesson_node_00007)
        # Smooths local temporal features before GRU
        self.conv = nn.Conv1d(
            HIDDEN_DIM,
            HIDDEN_DIM,
            kernel_size=CNN_KERNEL_SIZE,
            padding=CNN_KERNEL_SIZE // 2,
        )
        self.relu_conv = nn.ReLU()

        # 2. Recurrent Core (GRU)
        # We use batch_first=True because data loader returns (Batch, Time, Feats)
        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            bidirectional=BIDIRECTIONAL,
        )

        # Determine the output dimension of the GRU
        gru_output_dim = HIDDEN_DIM * (2 if BIDIRECTIONAL else 1)

        # 3. Classification Head
        # Maps GRU outputs to class logits for every frame
        self.classifier = nn.Linear(gru_output_dim, NUM_CLASSES)

    def forward(self, x, lengths):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, InputDim).
            lengths (torch.Tensor): 1D tensor containing the actual lengths of sequences.

        Returns:
            torch.Tensor: Logits of shape (Batch, Time, NumClasses).
        """
        # 1. Apply Feature Fusion
        # x shape: (B, T, InputDim) -> (B, T, HiddenDim)
        fused_features = self.feature_fusion(x)

        # 1.5 Apply Temporal Convolution
        # Permute to (B, Channel, Time) for Conv1d
        x_conv = fused_features.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = self.relu_conv(x_conv)
        # Permute back to (B, Time, Channel)
        x_conv = x_conv.permute(0, 2, 1)

        # 2. Pack Sequence for GRU
        # This handles variable length sequences efficiently and correctly ignores padding
        packed_input = pack_padded_sequence(
            x_conv, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 3. Pass through GRU
        # packed_output contains the hidden states for all time steps
        packed_output, _ = self.gru(packed_input)

        # 4. Unpack Sequence
        # rnn_output shape: (B, T, HiddenDim * NumDirections)
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 5. Classification Head
        # Apply linear layer to each time step
        # logits shape: (B, T, NumClasses)
        logits = self.classifier(rnn_output)

        return logits
