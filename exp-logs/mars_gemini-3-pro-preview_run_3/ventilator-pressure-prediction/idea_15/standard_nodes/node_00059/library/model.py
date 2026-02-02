import torch
import torch.nn as nn
from library.config import Config


class MultiKernelConvBlock(nn.Module):
    """
    A dense, multi-resolution convolutional block.
    Applies parallel 1D convolutions with different kernel sizes and concatenates the results.
    This allows the model to capture features at different local scales simultaneously
    (e.g., sharp changes vs smooth trends) without using dilation.
    """

    def __init__(self, in_channels, out_channels, kernels, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            # Padding is set to k // 2 to maintain temporal alignment (same padding)
            # Stride=1 and Dilation=1 ensure dense feature extraction
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    stride=1,
                    dilation=1,
                )
            )

        # The output dimension is the sum of the outputs of all parallel convolutions
        self.total_out_channels = out_channels * len(kernels)

        self.bn = nn.BatchNorm1d(self.total_out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        outputs = [conv(x) for conv in self.convs]
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FMDHNet(nn.Module):
    """
    Full-Scale Multi-Resolution Dense-Hybrid Network (FMDH-Net).

    Architecture:
    1. Parallel Branches:
       - CNN Branch: Multi-Kernel Dense Convolutions (Resistive dynamics, high freq).
       - LSTM Branch: High-Capacity Bidirectional LSTM (Elastic dynamics, integration).
    2. Fusion:
       - Concatenation of branch outputs.
       - Deep MLP Head.

    Constraints:
    - No additive skip connections between branches (forces flow through integrators).
    - Dense convolutions (no dilation) for maximum local fidelity.
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from Config
        input_dim = Config.INPUT_DIM

        # CNN Hyperparameters
        cnn_filters = Config.CNN_FILTERS
        cnn_kernels = Config.CNN_KERNELS
        cnn_dropout = Config.CNN_DROPOUT

        # LSTM Hyperparameters
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS
        lstm_bidir = Config.LSTM_BIDIRECTIONAL
        lstm_dropout = Config.LSTM_DROPOUT

        # Head Hyperparameters
        fc_dropout = Config.FC_DROPOUT

        # =====================================================================
        # Branch 1: Multi-Resolution Dense CNN (Resistive Stream)
        # =====================================================================

        # Initial projection to map input features to the CNN feature space
        self.cnn_projection = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size=1),
            nn.BatchNorm1d(cnn_filters),
            nn.GELU(),
        )

        # Stack of Multi-Kernel Blocks
        # We use a fixed depth of 4 blocks to ensure sufficient depth while relying on
        # the width (multi-kernel) for feature diversity.
        self.cnn_blocks = nn.ModuleList()

        # Block 1: Input (cnn_filters) -> Output (cnn_filters * num_kernels)
        self.cnn_blocks.append(
            MultiKernelConvBlock(cnn_filters, cnn_filters, cnn_kernels, cnn_dropout)
        )

        # Calculate the channel dimension after the first block
        # e.g., 64 * 3 = 192
        cnn_current_dim = cnn_filters * len(cnn_kernels)

        # Subsequent Blocks: Input (192) -> Output (192)
        # To maintain constant width, we set the 'out_channels' of the parallel convs
        # such that their sum equals the input dimension.
        # However, to maximize capacity as per "Critical Mass", we will keep the
        # individual kernel output size at 'cnn_filters', meaning the width stays at 192.
        # Block 1: In 64 -> Out 192
        # Block 2: In 192 -> Out 192 (Each conv takes 192, outputs 64)
        for _ in range(3):
            self.cnn_blocks.append(
                MultiKernelConvBlock(
                    cnn_current_dim, cnn_filters, cnn_kernels, cnn_dropout
                )
            )

        self.cnn_out_dim = cnn_current_dim

        # =====================================================================
        # Branch 2: High-Capacity BiLSTM (Elastic Stream)
        # =====================================================================
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidir,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        self.lstm_out_dim = lstm_hidden * 2 if lstm_bidir else lstm_hidden

        # =====================================================================
        # Fusion Head
        # =====================================================================
        fusion_dim = self.cnn_out_dim + self.lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Features)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # 1. Prepare Inputs
        # CNN requires (Batch, Channels, Seq_Len)
        x_cnn = x.transpose(1, 2)
        # LSTM requires (Batch, Seq_Len, Features)
        x_lstm = x

        # 2. TCN Branch Forward
        c = x_cnn
        for block in self.tcn_blocks:
            c = block(c)
        # Transpose back to (Batch, Seq_Len, Channels)
        c_out = c.transpose(1, 2)

        # 3. LSTM Branch Forward
        l_out, _ = self.lstm(x_lstm)

        # 4. Fusion
        # Concatenate features from both branches
        combined = torch.cat([c_out, l_out], dim=2)

        # 5. Prediction Head
        out = self.head(combined)

        return out
