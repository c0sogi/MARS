import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) block.
    Projects input to 2*dim, then applies GLU activation: x * sigmoid(gate).
    """

    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        # out: (Batch, Seq, Output_Dim)
        return F.glu(self.fc(x), dim=-1)


class DGC_BiLSTM(nn.Module):
    """
    Deep Gated-Cascade BiLSTM (DGC-BiLSTM) Model.

    Features:
    - Dual-Path Injection (Identity + GLU) for physics context retention.
    - Gated-Cascade Inter-Layer Connectivity (LayerNorm -> GLU -> Dropout).
    - Deep Recurrent Backbone (4-layer BiLSTM).
    """

    def __init__(self):
        super(DGC_BiLSTM, self).__init__()

        # Dimensions
        self.input_dim = 12  # Based on feature engineering in data_loader.py
        self.hidden_dim = Config.HIDDEN_DIM
        self.injection_dim = Config.INJECTION_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_p = Config.DROPOUT
        self.bidirectional = Config.BIDIRECTIONAL

        # LSTM Output Dimension (2x if bidirectional)
        self.lstm_out_dim = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )

        # ==========================================
        # 1. Bottlenecked Dual-Path Injection
        # ==========================================
        # Projects raw input to hidden space, applies residual GLU, then compresses
        self.stem_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.stem_glu = GLU(self.hidden_dim, self.hidden_dim)
        self.injection_bottleneck = nn.Linear(self.hidden_dim, self.injection_dim)

        # ==========================================
        # 2. Recurrent Backbone & Gated Cascade
        # ==========================================
        self.lstm_layers = nn.ModuleList()
        self.inter_layer_gates = nn.ModuleList()

        # Layer 0
        # Input: Raw Features + Injection Context
        self.lstm_layers.append(
            nn.LSTM(
                input_size=self.input_dim + self.injection_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        )

        # Layers 1 to N-1
        for _ in range(1, self.num_layers):
            # Inter-Layer Gating Block: Norm -> GLU -> Dropout
            self.inter_layer_gates.append(
                nn.Sequential(
                    nn.LayerNorm(self.lstm_out_dim),
                    GLU(self.lstm_out_dim, self.lstm_out_dim),
                    nn.Dropout(self.dropout_p),
                )
            )

            # LSTM Layer Input: Gated Previous Output + Injection Context
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=self.lstm_out_dim + self.injection_dim,
                    hidden_size=self.hidden_dim,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

        # ==========================================
        # 3. Prediction Head
        # ==========================================
        # Projects final LSTM output to scalar pressure
        self.head = nn.Linear(self.lstm_out_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Pressure predictions of shape (Batch, Seq_Len)
        """
        # 1. Compute Injection Context
        # Dual-Path: Identity + GLU
        stem = self.stem_proj(x)
        context = stem + self.stem_glu(stem)
        injection = self.injection_bottleneck(context)  # (Batch, Seq, Injection_Dim)

        # 2. Recurrent Propagation
        lstm_out = None

        for i, lstm in enumerate(self.lstm_layers):
            if i == 0:
                # First Layer: Cat(Raw, Injection)
                lstm_input = torch.cat([x, injection], dim=-1)
            else:
                # Subsequent Layers: Cat(Gated_Prev_Out, Injection)
                # Apply Gated-Cascade Block to previous output
                gated_prev = self.inter_layer_gates[i - 1](lstm_out)
                lstm_input = torch.cat([gated_prev, injection], dim=-1)

            # LSTM Forward
            # self.lstm returns (output, (h_n, c_n))
            lstm_out, _ = lstm(lstm_input)

        # 3. Head
        # lstm_out shape: (Batch, Seq, LSTM_Out_Dim)
        pred = self.head(lstm_out)  # (Batch, Seq, 1)

        return pred.squeeze(-1)
