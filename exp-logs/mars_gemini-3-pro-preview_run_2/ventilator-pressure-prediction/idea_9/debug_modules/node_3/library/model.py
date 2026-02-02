import torch
import torch.nn as nn
from library.config import Config


class GatedProjection(nn.Module):
    """
    Projects input features into a latent 'Physics Context' using a Gated Linear Unit (GLU).
    This creates a high-fidelity signal representation of the physical attributes (R, C)
    and control inputs, strictly excluding dropout to preserve signal determinism.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # GLU operation halves the dimension, so we project to 2 * output_dim
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim) -> (Batch, Seq, Output_Dim)
        return self.glu(self.linear(x))


class SC_GI_BiLSTM(nn.Module):
    """
    Skip-Context Gated-Injection BiLSTM (SC-GI-BiLSTM).

    Architecture Highlights:
    1. Physics Context: A gated projection of raw inputs used as a global conditioning signal.
    2. Input Injection: The context is concatenated to the input of *every* LSTM layer,
       ensuring deep layers retain access to lung attributes (R, C).
    3. Skip-Context Readout: The context is concatenated with the final LSTM output
       before the regression head, allowing direct modeling of instantaneous resistive pressure.
    """

    def __init__(
        self,
        input_dim=None,
        lstm_hidden_dim=None,
        projection_dim=None,
        lstm_layers=None,
        dropout=None,
        bidirectional=None,
    ):
        super().__init__()

        self.input_dim = input_dim if input_dim is not None else Config.INPUT_DIM
        self.hidden_dim = (
            lstm_hidden_dim if lstm_hidden_dim is not None else Config.LSTM_HIDDEN_DIM
        )
        self.proj_dim = (
            projection_dim if projection_dim is not None else Config.PROJECTION_DIM
        )
        self.num_layers = lstm_layers if lstm_layers is not None else Config.LSTM_LAYERS
        self.bidirectional = (
            bidirectional if bidirectional is not None else Config.BIDIRECTIONAL
        )
        self.num_directions = 2 if bidirectional else 1

        # --- 1. Gated Latent Projection ---
        # Projects raw features into the 'Physics Context'
        self.context_proj = GatedProjection(self.input_dim, self.proj_dim)

        # --- 2. Deep Recurrent Backbone with Input Injection ---
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(self.num_layers):
            # Calculate input dimension for the current LSTM layer
            if i == 0:
                # First layer: Raw Input + Context
                layer_in_dim = self.input_dim + self.proj_dim
            else:
                # Subsequent layers: Previous Layer Output + Context
                # Previous output dim is hidden_dim * num_directions
                layer_in_dim = (self.hidden_dim * self.num_directions) + self.proj_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_in_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            # Layer Normalization is applied after every recurrent block
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim * self.num_directions))

        # --- 3. Skip-Context Readout ---
        # The regression head receives the Final LSTM Output concatenated with the Context.
        # This allows explicit modeling of the resistive component (R * Flow) via the Context,
        # while the LSTM models the elastic component (Volume / C).
        readout_in_dim = (self.hidden_dim * self.num_directions) + self.proj_dim
        self.head = nn.Linear(readout_in_dim, 1)

    def forward(self, x):
        """
        Forward pass of the SC-GI-BiLSTM.

        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Seq_Len, Features).

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len).
        """
        # 1. Generate Physics Context
        # context: (Batch, Seq, Proj_Dim)
        context = self.context_proj(x)

        # 2. Recurrent Layers with Input Injection
        current_input = x

        for i in range(self.num_layers):
            # Concatenate current input (or previous hidden state) with the Context
            # shape: (Batch, Seq, Layer_In_Dim)
            lstm_input = torch.cat([current_input, context], dim=-1)

            # Pass through LSTM layer
            # output: (Batch, Seq, Hidden * Dirs)
            output, _ = self.lstm_layers[i](lstm_input)

            # Apply Layer Normalization
            output = self.layer_norms[i](output)

            # Update current_input for the next layer
            current_input = output

        # 3. Skip-Context Readout
        # Concatenate the final LSTM output with the original Physics Context
        # shape: (Batch, Seq, Hidden * Dirs + Proj_Dim)
        readout_input = torch.cat([current_input, context], dim=-1)

        # Project to scalar pressure
        # shape: (Batch, Seq, 1)
        prediction = self.head(readout_input)

        # Remove the feature dimension to return (Batch, Seq)
        return prediction.squeeze(-1)
