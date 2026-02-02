import torch
import torch.nn as nn
from library.config import Config


class GatedProjection(nn.Module):
    """
    Projects raw features into a high-dimensional latent space using a Gated Linear Unit (GLU).
    This serves as the deterministic 'Physics Context' generator.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # GLU requires the input feature dimension to be 2 * output dimension
        self.linear = nn.Linear(input_dim, hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        # linear out: (Batch, Seq, 2 * Hidden_Dim)
        # glu out: (Batch, Seq, Hidden_Dim)
        return self.glu(self.linear(x))


class RGIBiLSTM(nn.Module):
    """
    Regularized Gated-Injection BiLSTM (RGI-BiLSTM).

    Key Components:
    - Gated Latent Projection: Creates a high-fidelity physics context.
    - Deep Recurrent Backbone: 4 layers of BiLSTMs.
    - Input Injection: Physics context is concatenated to the input of EVERY LSTM layer.
    - Internal Regularization: LayerNorm and Dropout applied after each LSTM layer.
    """

    def __init__(self):
        super().__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_p = Config.DROPOUT

        # 1. Physics Context Projection
        self.projection = GatedProjection(self.input_dim, self.hidden_dim)

        # 2. Recurrent Backbone
        self.lstm_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Calculate input dimension for this layer
            if i == 0:
                # First layer: Raw Features + Context
                layer_input_dim = self.input_dim + self.hidden_dim
            else:
                # Subsequent layers: Previous Layer Output (BiDir) + Context
                # BiDir output is 2 * hidden_dim
                layer_input_dim = (2 * self.hidden_dim) + self.hidden_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Regularization after LSTM
            self.norms.append(nn.LayerNorm(2 * self.hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout_p))

        # 3. Readout Head
        # Takes the output of the final BiLSTM layer
        self.head = nn.Linear(2 * self.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stability."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # Generate Physics Context
        context = self.projection(x)  # (Batch, Seq, Hidden_Dim)

        current_input = x

        for i in range(self.num_layers):
            # Input Injection: Concatenate context to the input of the layer
            combined_input = torch.cat([current_input, context], dim=-1)

            # LSTM Forward
            self.lstm_layers[i].flatten_parameters()
            lstm_out, _ = self.lstm_layers[i](combined_input)

            # Regularization
            lstm_out = self.norms[i](lstm_out)
            lstm_out = self.dropouts[i](lstm_out)

            # Set input for next layer
            current_input = lstm_out

        # Final Prediction
        pred = self.head(current_input)  # (Batch, Seq, 1)

        return pred.squeeze(-1)  # (Batch, Seq)
