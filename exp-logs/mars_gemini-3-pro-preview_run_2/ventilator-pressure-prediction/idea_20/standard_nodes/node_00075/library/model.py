import torch
import torch.nn as nn
from library.config import Config


class WideGLU(nn.Module):
    """
    Wide Monolithic Context Extractor using Gated Linear Unit.
    Projects input to 2x size, splits into content/gate, and applies sigmoid gating.
    """

    def __init__(self, input_dim, output_dim):
        super(WideGLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        # Project
        out = self.fc(x)
        # Split into content and gate
        content, gate = out.chunk(2, dim=-1)
        # Gating mechanism
        return content * torch.sigmoid(gate)


class CWCDP_BiLSTM(nn.Module):
    """
    Corrected Wide-Context Dual-Path BiLSTM (CWCDP-BiLSTM).

    Features:
    - Wide Monolithic GLU for context extraction.
    - Dual-Path Injection Payload (Identity + Context) fed to ALL layers.
    - Deep Recurrent Backbone with Inter-layer Norm and Dropout.
    """

    def __init__(self):
        super(CWCDP_BiLSTM, self).__init__()

        # Hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.glu_size = Config.GLU_SIZE
        self.hidden_size = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_LAYERS
        self.bidirectional = Config.BIDIRECTIONAL
        self.dropout_rate = Config.DROPOUT

        # 1. Wide Monolithic Context Extractor
        self.glu = WideGLU(self.input_dim, self.glu_size)

        # Calculate Injection Payload Size: Raw Input (Identity) + GLU Context
        self.payload_size = self.input_dim + self.glu_size

        # 2. Wide Deep Recurrent Backbone
        # We use ModuleList to handle the custom Deep Injection logic manually
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Calculate LSTM output size
        self.lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        for i in range(self.num_layers):
            # Input Logic for Deep Injection:
            # Layer 0: Input is just the Injection Payload
            # Layer N: Input is (Previous Layer Output) + (Injection Payload)
            if i == 0:
                layer_input_size = self.payload_size
            else:
                layer_input_size = self.lstm_output_size + self.payload_size

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )
            # Layer Normalization for stability between recurrent layers
            self.layer_norms.append(nn.LayerNorm(self.lstm_output_size))

        # Inter-layer Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. Regression Head
        self.head = nn.Linear(self.lstm_output_size, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Pressure predictions of shape (Batch, Seq_Len)
        """
        # 1. Context Extraction
        context = self.glu(x)

        # 2. Construct Injection Payload (Identity + Context)
        # No dropout applied to the payload itself to preserve signal fidelity
        payload = torch.cat([x, context], dim=-1)

        # Initialize loop variables
        curr_input = payload
        prev_output = None

        # 3. Deep Recurrent Loop
        for i in range(self.num_layers):
            # Deep Injection Logic
            if i > 0:
                # Concatenate previous layer output with the stable payload
                curr_input = torch.cat([prev_output, payload], dim=-1)

            # LSTM Forward
            # output shape: (Batch, Seq, Hidden*2)
            output, _ = self.lstm_layers[i](curr_input)

            # Apply Inter-Layer Connectivity (Norm -> Dropout)
            output = self.layer_norms[i](output)
            output = self.dropout(output)

            # Update previous output for next layer
            prev_output = output

        # 4. Final Prediction
        # prev_output contains the sequence output of the last LSTM layer
        pred = self.head(prev_output)

        # Squeeze the last dimension to match target shape (Batch, Seq)
        return pred.squeeze(-1)
