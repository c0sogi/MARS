import torch
import torch.nn as nn
from library.config import Config


class MultiScaleConvBlock(nn.Module):
    """
    Branch 1: Multi-Scale Dense CNN (The Resistive Stream).
    Operates parallel convolutions with different kernel sizes to capture
    dynamics at multiple temporal scales simultaneously.
    """

    def __init__(self, input_dim, filters, kernels):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList()

        for k in kernels:
            # Dilation=1 (Dense), Padding='same' (k//2 for odd kernels)
            # This ensures the output length matches the input length without phase shift
            self.convs.append(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=filters,
                    kernel_size=k,
                    padding=k // 2,
                    dilation=1,
                )
            )

        self.activation = nn.GELU()

    def forward(self, x):
        # Input x: (Batch, Seq_Len, Input_Dim)
        # Conv1d expects: (Batch, Input_Dim, Seq_Len)
        x = x.transpose(1, 2)

        outputs = []
        for conv in self.convs:
            out = self.activation(conv(x))
            outputs.append(out)

        # Concatenate along the channel dimension (dim=1)
        # Shape becomes: (Batch, Filters * Num_Kernels, Seq_Len)
        out = torch.cat(outputs, dim=1)

        # Transpose back to (Batch, Seq_Len, Features)
        out = out.transpose(1, 2)
        return out


class MSDHNet(nn.Module):
    """
    Multi-Scale Dense-Hybrid Network (MSDH-Net).
    Combines a Multi-Scale CNN branch for high-frequency resistive dynamics
    and a high-capacity Bidirectional LSTM branch for low-frequency elastic dynamics.
    """

    def __init__(self):
        super(MSDHNet, self).__init__()

        # ---------------------------------------------------------------------
        # Hyperparameters from Config
        # ---------------------------------------------------------------------
        input_dim = Config.INPUT_DIM

        # CNN Branch Config
        cnn_kernels = Config.CNN_KERNELS
        cnn_filters = Config.CNN_FILTERS

        # LSTM Branch Config
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS
        lstm_bidir = Config.LSTM_BIDIRECTIONAL
        lstm_dropout = Config.LSTM_DROPOUT

        # Fusion Head Config
        fc_hidden = Config.FC_HIDDEN_UNITS
        fc_dropout = Config.FC_DROPOUT

        # ---------------------------------------------------------------------
        # Architecture Definition
        # ---------------------------------------------------------------------

        # Branch 1: Multi-Scale Dense CNN
        self.cnn_branch = MultiScaleConvBlock(input_dim, cnn_filters, cnn_kernels)
        self.cnn_out_dim = len(cnn_kernels) * cnn_filters

        # Branch 2: High-Capacity Bidirectional LSTM
        # Serves as numerical integrator
        self.lstm_branch = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidir,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_out_dim = lstm_hidden * 2 if lstm_bidir else lstm_hidden

        # Fusion Head: Coupled Latent Integration
        # Concatenates CNN and LSTM outputs -> Dense MLP
        fusion_input_dim = self.cnn_out_dim + self.lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len)
        """
        # 1. Branch 1: Resistive Stream (CNN)
        # Captures local gradients and high-frequency dynamics
        cnn_out = self.cnn_branch(x)

        # 2. Branch 2: Elastic Stream (LSTM)
        # Captures integral terms and long-range dependencies
        lstm_out, _ = self.lstm_branch(x)

        # 3. Fusion
        # Concatenate features from both branches
        combined = torch.cat([cnn_out, lstm_out], dim=-1)

        # 4. Prediction Head
        # Projects to scalar pressure
        out = self.head(combined)

        # Remove the last dimension to match target shape (Batch, Seq_Len)
        return out.squeeze(-1)
