import torch
import torch.nn as nn
from library.config import Config


class DualPathInjection(nn.Module):
    """
    Uncompressed Dual-Path Injection Block.

    Path A: Identity (Explicit Fidelity) - Raw features + Physics terms.
    Path B: Wide Monolithic GLU (Wide Context) - High-bandwidth non-linear context.

    The outputs are concatenated to form a stable ground truth payload.
    Dropout is strictly excluded from this block.
    """

    def __init__(self, input_dim, glu_hidden_dim):
        super().__init__()
        # GLU requires 2 * hidden_dim output from the linear layer to split into value and gate
        self.glu_fc = nn.Linear(input_dim, glu_hidden_dim * 2)
        self.act = nn.GLU(dim=-1)

    def forward(self, x):
        # Path A: Identity
        path_a = x

        # Path B: Wide Context
        # x: (batch, seq, input_dim) -> (batch, seq, glu_hidden_dim)
        path_b = self.act(self.glu_fc(x))

        # Fusion: Concatenate raw physics features with learned context
        return torch.cat([path_a, path_b], dim=-1)


class WPABiLSTM(nn.Module):
    """
    Wide-Bandwidth Physics-Augmented BiLSTM.

    Features:
    - Deep Injection: The payload from DualPathInjection is fed to every LSTM layer.
    - Wide Backbone: 512 hidden units per direction.
    - Stretched Depth: 4 layers with LayerNorm and Dropout.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.glu_dim = Config.GLU_HIDDEN_SIZE
        self.hidden_size = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_LAYERS
        self.bidirectional = Config.BIDIRECTIONAL
        self.dropout_prob = Config.DROPOUT

        # 1. Injection Block
        self.injector = DualPathInjection(self.input_dim, self.glu_dim)

        # Calculate dimension of the injection payload
        # Payload = Raw Input + GLU Output
        self.payload_dim = self.input_dim + self.glu_dim

        # 2. Recurrent Backbone
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Calculate LSTM output dimension
        self.lstm_output_dim = (
            self.hidden_size * 2 if self.bidirectional else self.hidden_size
        )

        for i in range(self.num_layers):
            # Input size logic for Deep Injection:
            # Layer 0: Receives just the payload.
            # Layer >0: Receives previous layer output concatenated with the payload.
            if i == 0:
                input_size = self.payload_dim
            else:
                input_size = self.lstm_output_dim + self.payload_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            # Stabilization between layers
            self.layer_norms.append(nn.LayerNorm(self.lstm_output_dim))
            self.dropouts.append(nn.Dropout(self.dropout_prob))

        # 3. Regression Head
        self.head = nn.Linear(self.lstm_output_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            predictions: Tensor of shape (batch_size, seq_len)
        """
        # Generate the Wide-Bandwidth Payload
        # Shape: (batch, seq, payload_dim)
        payload = self.injector(x)

        current_input = payload

        for i in range(self.num_layers):
            # Pass through LSTM layer
            output, _ = self.lstm_layers[i](current_input)

            # Apply Normalization and Dropout
            output = self.layer_norms[i](output)
            output = self.dropouts[i](output)

            # Prepare input for the next layer (Deep Injection)
            if i < self.num_layers - 1:
                # Concatenate current output with the original payload
                current_input = torch.cat([output, payload], dim=-1)
            else:
                # For the final layer, the output goes to the head
                current_input = output

        # Project to scalar pressure
        # Shape: (batch, seq, 1)
        pred = self.head(current_input)

        # Squeeze to (batch, seq) to match target shape
        return pred.squeeze(-1)
