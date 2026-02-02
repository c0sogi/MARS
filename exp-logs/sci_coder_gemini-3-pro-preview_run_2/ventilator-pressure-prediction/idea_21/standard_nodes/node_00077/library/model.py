import torch
import torch.nn as nn
from library.config import Config


class WideGLU(nn.Module):
    """
    Wide Monolithic Gated Linear Unit (GLU).
    Allows the model to learn cross-term interactions immediately from the full feature set.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Project to 2x hidden_dim to create content (A) and gate (B) branches
        self.fc = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x):
        # x shape: (Batch, Seq, Input_Dim)
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        # GLU operation: A * sigmoid(B)
        return a * torch.sigmoid(b)


class FPBC_BiLSTM(nn.Module):
    """
    Fidelity-Preserving Bottlenecked-Context BiLSTM.

    Architecture:
    1. Context Extractor: Wide GLU -> Linear Bottleneck.
    2. Injection Payload: Concatenation of Raw Input (Fidelity) + Context (Abstraction).
    3. Backbone: 4-layer BiLSTM with Deep Input Injection (Payload fed to every layer).
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        input_dim = Config.INPUT_DIM
        glu_size = Config.GLU_WIDE_SIZE
        bottleneck_size = Config.CONTEXT_BOTTLENECK_SIZE
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        num_layers = Config.LSTM_LAYERS
        dropout_p = Config.MODEL_DROPOUT

        # ---------------------------------------------------------
        # 1. Monolithic Context Extractor
        # ---------------------------------------------------------
        self.glu = WideGLU(input_dim, glu_size)
        self.bottleneck = nn.Linear(glu_size, bottleneck_size)

        # ---------------------------------------------------------
        # 2. Fidelity-Preserving Payload Definition
        # ---------------------------------------------------------
        # Payload = Raw Features (Identity) + Compressed Context
        self.payload_size = input_dim + bottleneck_size

        # ---------------------------------------------------------
        # 3. Wide Deep Recurrent Backbone
        # ---------------------------------------------------------
        # We use ModuleList to allow manual injection of the payload at every step.
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)

        for i in range(num_layers):
            # Calculate input size for this specific layer
            if i == 0:
                # First layer takes just the payload
                layer_input_dim = self.payload_size
            else:
                # Subsequent layers take: Previous Output + Payload
                # Previous output is bidirectional, so size is lstm_hidden * 2
                layer_input_dim = (lstm_hidden * 2) + self.payload_size

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=lstm_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Layer Normalization applied to the output of the BiLSTM
            self.layer_norms.append(nn.LayerNorm(lstm_hidden * 2))

        # ---------------------------------------------------------
        # 4. Regression Head
        # ---------------------------------------------------------
        self.head = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x, u_out=None):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq, Features).
            u_out (torch.Tensor, optional): Control input (Batch, Seq). Not used in architecture.

        Returns:
            torch.Tensor: Predicted pressure (Batch, Seq).
        """
        # ---------------------------------------------------------
        # A. Context Generation
        # ---------------------------------------------------------
        # Extract wide features
        glu_out = self.glu(x)
        # Compress to bottleneck
        context = self.bottleneck(glu_out)

        # ---------------------------------------------------------
        # B. Payload Construction
        # ---------------------------------------------------------
        # Concatenate Raw Input (Fidelity) and Context (Abstract)
        # Strictly no dropout here to maintain signal stability
        payload = torch.cat([x, context], dim=-1)

        # ---------------------------------------------------------
        # C. Deep Injection Recurrent Pass
        # ---------------------------------------------------------
        current_input = payload

        for i, lstm in enumerate(self.lstm_layers):
            # Pass through LSTM layer
            lstm_out, _ = lstm(current_input)

            # Apply Inter-Layer Regularization
            lstm_out = self.layer_norms[i](lstm_out)
            lstm_out = self.dropout(lstm_out)

            # Prepare input for the next layer
            if i < len(self.lstm_layers) - 1:
                # Deep Injection: Concatenate current output with the persistent payload
                current_input = torch.cat([lstm_out, payload], dim=-1)
            else:
                # Final layer output
                current_input = lstm_out

        # ---------------------------------------------------------
        # D. Prediction
        # ---------------------------------------------------------
        # Project to scalar pressure
        pressure = self.head(current_input)

        # Squeeze the last dimension (Batch, Seq, 1) -> (Batch, Seq)
        return pressure.squeeze(-1)
