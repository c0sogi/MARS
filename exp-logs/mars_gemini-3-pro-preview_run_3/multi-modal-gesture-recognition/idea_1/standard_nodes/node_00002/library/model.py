import torch
import torch.nn as nn
from library.config import Config


class BiGRUModel(nn.Module):
    """
    Bi-directional GRU Network for Frame-wise Gesture Recognition.

    Architecture:
    1. Input Projection (Linear + ReLU)
    2. Bi-directional GRU (Multi-layer)
    3. Dropout
    4. Classification Head (Linear)
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ):
        """
        Args:
            input_dim (int): Dimension of input features (Skeleton + Audio).
            hidden_dim (int): Dimension of GRU hidden states.
            num_layers (int): Number of stacked GRU layers.
            num_classes (int): Number of output classes (Gestures + Background).
            dropout (float): Dropout probability.
        """
        super(BiGRUModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Projection layer to map input features to hidden dimension
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Bi-directional GRU
        # batch_first=True expects input shape (Batch, SeqLen, Features)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Classification head
        # Input to this layer is hidden_dim * 2 because of bidirectionality
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Padded input sequences. Shape (Batch, MaxLen, InputDim).
            lengths (torch.Tensor): Actual lengths of sequences. Shape (Batch,).

        Returns:
            torch.Tensor: Logits for each frame. Shape (Batch, MaxLen, NumClasses).
        """
        # 1. Project features
        # x shape: (Batch, MaxLen, InputDim) -> (Batch, MaxLen, HiddenDim)
        x = self.projection(x)

        # 2. Pack sequence for RNN
        # lengths must be on CPU for pack_padded_sequence in some versions,
        # ensuring robustness by moving to cpu.
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 3. Pass through GRU
        # packed_output shape: (TotalFrames, HiddenDim * 2)
        packed_output, _ = self.gru(packed_input)

        # 4. Unpack sequence
        # output shape: (Batch, MaxLen, HiddenDim * 2)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 5. Apply Dropout
        output = self.dropout(output)

        # 6. Classification Head
        # logits shape: (Batch, MaxLen, NumClasses)
        logits = self.classifier(output)

        return logits
