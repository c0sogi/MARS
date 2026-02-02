import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Inception-style).
    Applies parallel 1D convolutions with different kernel sizes to capture
    features at various temporal resolutions.
    """

    def __init__(self, input_dim, filters, kernels):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            # Padding is 'same' to preserve sequence length
            # padding = (kernel_size - 1) // 2 for odd kernels
            pad = (k - 1) // 2
            self.convs.append(nn.Conv1d(input_dim, filters, kernel_size=k, padding=pad))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        # Conv1d expects: (Batch, Input_Dim, Seq_Len)
        x = x.transpose(1, 2)

        outputs = []
        for conv in self.convs:
            out = F.relu(conv(x))
            outputs.append(out)

        # Concatenate along the channel dimension (dim=1)
        x_cat = torch.cat(outputs, dim=1)

        # Transpose back to (Batch, Seq_Len, Output_Dim)
        x_cat = x_cat.transpose(1, 2)
        return x_cat


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for global context gating.
    Computes a global context vector via Global Average Pooling and an MLP,
    then rescales the input features.
    """

    def __init__(self, input_dim, reduction_ratio=16):
        super().__init__()
        reduced_dim = max(1, input_dim // reduction_ratio)
        self.fc1 = nn.Linear(input_dim, reduced_dim)
        self.fc2 = nn.Linear(reduced_dim, input_dim)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # Global Average Pooling over the time dimension (dim=1)
        # Result: (Batch, Input_Dim)
        y = torch.mean(x, dim=1)

        # MLP
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Reshape for broadcasting: (Batch, 1, Input_Dim)
        y = y.unsqueeze(1)

        return y


class DualGatedResidualBlock(nn.Module):
    """
    Dual-Gated Residual Bi-LSTM Block.
    Combines LSTM recurrence with:
    1. Local Gating (Pointwise)
    2. Global Gating (SE-Block)
    3. Residual Connection with Projection
    """

    def __init__(
        self, input_dim, hidden_dim, bidirectional=True, dropout=0.0, se_ratio=16
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_dim * self.num_directions

        # Recurrence: LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Local Gating: Pointwise Linear -> Sigmoid
        self.local_gate = nn.Linear(self.lstm_output_dim, self.lstm_output_dim)

        # Global Gating: SE Block
        self.global_gate = SEBlock(self.lstm_output_dim, reduction_ratio=se_ratio)

        # Projection Shortcut: If input dim != output dim, project input
        if input_dim != self.lstm_output_dim:
            self.projection = nn.Linear(input_dim, self.lstm_output_dim)
        else:
            self.projection = nn.Identity()

        # Dropout applied to the residual branch
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # 1. Recurrence
        h, _ = self.lstm(x)  # h shape: (Batch, Seq_Len, LSTM_Output_Dim)

        # 2. Local Gating
        g_local = torch.sigmoid(self.local_gate(h))

        # 3. Global Gating
        g_global = self.global_gate(
            h
        )  # Broadcasts to (Batch, Seq_Len, LSTM_Output_Dim)

        # 4. Fusion
        h_prime = h * g_local * g_global

        # 5. Residual Connection
        # Project input to match dimensions if necessary
        x_proj = self.projection(x)

        # Add dropout to the branch before addition
        out = x_proj + self.dropout(h_prime)

        return out


class VentilatorNet(nn.Module):
    """
    Dual-Gated Multi-Scale CNN-LSTM Network.
    Structure:
    1. Stem: Multi-Scale CNN
    2. Backbone: Stack of Dual-Gated Residual Bi-LSTM Blocks
    3. Head: Linear Projection to Pressure
    """

    def __init__(self, input_dim):
        super().__init__()

        # --- Stem ---
        self.stem = MultiScaleCNN(
            input_dim=input_dim, filters=Config.CNN_FILTERS, kernels=Config.CNN_KERNELS
        )

        # Calculate Stem Output Dimension
        # Concatenation of N kernels * filters
        stem_out_dim = len(Config.CNN_KERNELS) * Config.CNN_FILTERS

        # --- Backbone ---
        layers = []
        current_dim = stem_out_dim

        for _ in range(Config.LSTM_LAYERS):
            layers.append(
                DualGatedResidualBlock(
                    input_dim=current_dim,
                    hidden_dim=Config.LSTM_HIDDEN_DIM,
                    bidirectional=Config.BIDIRECTIONAL,
                    dropout=Config.DROPOUT,
                    se_ratio=Config.SE_RATIO,
                )
            )
            # Update current_dim for the next layer
            # The output of the block is hidden_dim * num_directions
            current_dim = Config.LSTM_HIDDEN_DIM * (2 if Config.BIDIRECTIONAL else 1)

        self.backbone = nn.Sequential(*layers)

        # --- Head ---
        # Simple Linear layer to project to scalar pressure
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # Stem
        x = self.stem(x)

        # Backbone
        x = self.backbone(x)

        # Head
        # Project to (Batch, Seq_Len, 1)
        out = self.head(x)

        # Remove last dimension to get (Batch, Seq_Len)
        return out.squeeze(-1)
