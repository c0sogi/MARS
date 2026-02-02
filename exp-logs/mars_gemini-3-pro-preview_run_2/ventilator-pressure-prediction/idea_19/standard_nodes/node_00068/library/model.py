import torch
import torch.nn as nn
from library.config import Config


class WideGLU(nn.Module):
    """
    Wide Monolithic Gated Linear Unit (GLU) for context extraction.
    Projects input features to a high-dimensional space and applies gating
    to learn complex interactions (e.g., R * Flow) immediately.
    """

    def __init__(self, input_dim, hidden_dim):
        super(WideGLU, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x):
        # Project to 2 * hidden_dim
        out = self.fc(x)
        # Split into signal and gate
        a, b = out.chunk(2, dim=-1)
        # Gating mechanism
        return a * torch.sigmoid(b)


class CWCDP_BiLSTM(nn.Module):
    """
    Corrected Wide-Context Dual-Path BiLSTM (CWCDP-BiLSTM).

    Architecture:
    1. Wide Monolithic Context Extractor (GLU).
    2. Dual-Path Injection Payload: Concatenation of Raw Input (Identity) and Context.
    3. Wide Deep Recurrent Backbone: 4-layer BiLSTM (512 units).
    4. Deep Injection: Payload is concatenated to the input of EVERY LSTM layer.
    """

    def __init__(self):
        super(CWCDP_BiLSTM, self).__init__()

        # Hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.glu_hidden = Config.GLU_HIDDEN_SIZE
        self.lstm_hidden = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_LAYERS
        self.dropout_prob = Config.DROPOUT

        # 1. Wide Monolithic Context Extractor
        self.glu = WideGLU(self.input_dim, self.glu_hidden)

        # Define Injection Payload Dimension
        # Payload = Raw Features (Path A) + GLU Context (Path B)
        self.payload_dim = self.input_dim + self.glu_hidden

        # 2. Deep Recurrent Backbone with Deep Injection
        # We use ModuleList to manually handle the concatenation at each step
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Determine input size for this layer
            if i == 0:
                # First layer takes just the injection payload
                layer_input_size = self.payload_dim
            else:
                # Subsequent layers take: Previous Output + Injection Payload
                # Previous output size = lstm_hidden * 2 (Bidirectional)
                layer_input_size = (self.lstm_hidden * 2) + self.payload_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=self.lstm_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Inter-layer regularization (not applied after the last layer)
            if i < self.num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(self.lstm_hidden * 2))
                self.dropouts.append(nn.Dropout(self.dropout_prob))

        # 3. Output Head
        # Projects final LSTM output (bidirectional) to scalar pressure
        self.head = nn.Linear(self.lstm_hidden * 2, 1)

    def forward(self, x, u_out=None):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq, Input_Dim)
            u_out (torch.Tensor, optional): Binary control input. Not used for prediction
                                            logic but kept for signature compatibility.
        Returns:
            torch.Tensor: Predicted pressure (Batch, Seq, 1)
        """
        # 1. Context Extraction
        context = self.glu(x)

        # 2. Construct Injection Payload (Dual-Path)
        # Path A: Raw Input (Signal Fidelity)
        # Path B: Context (Physics Context)
        # No dropout here to maintain ground truth stability
        payload = torch.cat([x, context], dim=-1)

        curr_input = payload

        # 3. Deep Recurrent Loop
        final_out = None

        for i in range(self.num_layers):
            # Pass through LSTM layer
            lstm_out, _ = self.lstm_layers[i](curr_input)

            if i < self.num_layers - 1:
                # Apply Norm and Dropout between layers
                lstm_out = self.layer_norms[i](lstm_out)
                lstm_out = self.dropouts[i](lstm_out)

                # Deep Injection: Concatenate payload to prepare input for next layer
                curr_input = torch.cat([lstm_out, payload], dim=-1)
            else:
                final_out = lstm_out

        # 4. Projection
        pred = self.head(final_out)

        return pred
