import torch
import torch.nn as nn
from library.config import Config


class GatedProjection(nn.Module):
    """
    Gated Linear Unit (GLU) projection block.
    Projects raw features into a latent space with a learnable gating mechanism.
    Strictly deterministic (no dropout) to serve as a stable anchor for deep layers.
    """

    def __init__(self, input_dim, output_dim):
        super(GatedProjection, self).__init__()
        self.linear_val = nn.Linear(input_dim, output_dim)
        self.linear_gate = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        val = self.linear_val(x)
        gate = self.sigmoid(self.linear_gate(x))
        # Element-wise multiplication (Hadamard product)
        return val * gate


class GIDBiLSTM(nn.Module):
    """
    Gated-Injection Deep Bidirectional LSTM (GI-DBiLSTM).

    Architecture:
    1. Gated Projection of input features.
    2. Stack of Bidirectional LSTM layers.
    3. Deep Injection: The gated projection is concatenated to the input of
       EVERY LSTM layer, not just the first.
    4. Layer Normalization after every LSTM block.
    """

    def __init__(self):
        super(GIDBiLSTM, self).__init__()

        # Load hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.hidden_size = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_LAYERS
        self.glu_dim = Config.GLU_DIM
        self.bidirectional = Config.BIDIRECTIONAL
        self.dropout_prob = Config.LSTM_DROPOUT

        # 1. Gated Latent Projection
        self.glu = GatedProjection(self.input_dim, self.glu_dim)

        # 2. Recurrent Backbone
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Direction multiplier
        dirs = 2 if self.bidirectional else 1
        self.hidden_output_size = self.hidden_size * dirs

        for i in range(self.num_layers):
            # Determine input size for this layer
            if i == 0:
                # First layer receives only the GLU output
                layer_input_size = self.glu_dim
            else:
                # Subsequent layers receive: Previous Output + GLU Injection
                layer_input_size = self.hidden_output_size + self.glu_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,  # Stacked manually to allow injection
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            # Layer Normalization stabilizes hidden state dynamics
            self.layer_norms.append(nn.LayerNorm(self.hidden_output_size))

        # Dropout for regularization between LSTM layers
        self.dropout = nn.Dropout(self.dropout_prob)

        # 3. Regression Head
        self.head = nn.Linear(self.hidden_output_size, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len)
        """
        # 1. Generate Gated Context Signal
        # Shape: (Batch, Seq, GLU_Dim)
        context = self.glu(x)

        current_input = context

        # 2. Pass through Deep Recurrent Stack
        for i, lstm in enumerate(self.lstm_layers):
            # Injection Logic
            if i > 0:
                # Concatenate previous output with the original gated context
                # current_input was the output of the previous layer
                # context is the stable physical signal
                current_input = torch.cat([current_input, context], dim=-1)

            # LSTM Forward Pass
            # output shape: (Batch, Seq, Hidden * Dirs)
            output, _ = lstm(current_input)

            # Stabilization
            output = self.layer_norms[i](output)

            # Regularization (apply dropout between layers, but not after the last one)
            if i < self.num_layers - 1:
                output = self.dropout(output)

            # Update input for next layer
            current_input = output

        # 3. Final Prediction
        # Shape: (Batch, Seq, 1)
        pred = self.head(current_input)

        # Squeeze to (Batch, Seq) to match target shape
        return pred.squeeze(-1)
