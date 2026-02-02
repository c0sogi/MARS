import torch
import torch.nn as nn


class ResBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        """
        Deep Residual Bidirectional LSTM (Res-BiLSTM) architecture.

        Implements a deep recurrent network where the input projection is concatenated
        to the input of every LSTM layer (Dense Residual Connection) to preserve
        signal fidelity of physical attributes (R, C, u_in) throughout the depth of the network.

        Cite solution_lesson_node_00021: Input Injection is retained.
        Added LayerNorm and Dropout for regularization.

        Args:
            input_dim (int): Number of input features from the dataset.
            hidden_dim (int): Dimension of the latent feature space and LSTM hidden states.
            num_layers (int): Number of stacked LSTM layers.
            output_dim (int): Dimension of the output (1 for pressure).
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input Projection: Projects engineered features to high-dimensional latent space.
        # Includes activation to initialize non-linear feature extraction.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )

        # Recurrent Backbone
        self.lstm_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            # Determine input size for the current LSTM layer
            if i == 0:
                # The first layer takes the projected input directly.
                layer_in_dim = hidden_dim
            else:
                # Subsequent layers implement the Dense Residual Connection.
                # Input = Concat(Output of Previous Layer, Original Projection)
                # Previous Output is Bidirectional -> 2 * hidden_dim
                # Original Projection -> hidden_dim
                layer_in_dim = (2 * hidden_dim) + hidden_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_in_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            # LayerNorm for the output of the BiLSTM (2 * hidden_dim)
            self.norms.append(nn.LayerNorm(2 * hidden_dim))

        # Regression Head
        # Maps the final bidirectional output to the target pressure.
        self.head = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim).

        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len, Output_Dim).
        """
        # 1. Project inputs to latent space
        # x_proj shape: (Batch, Seq_Len, Hidden_Dim)
        x_proj = self.projection(x)

        # Initialize the hidden representation with the projection
        h = x_proj

        # 2. Pass through Deep LSTM stack with Dense Residual Connections
        for i, (layer, norm) in enumerate(zip(self.lstm_layers, self.norms)):
            if i > 0:
                # For layers > 0, concatenate the original projection to the current hidden representation.
                # This ensures the deep layers still have direct access to the physical signal.
                # h shape before concat: (Batch, Seq_Len, 2 * Hidden_Dim) (from previous BiLSTM)
                # x_proj shape: (Batch, Seq_Len, Hidden_Dim)
                # Resulting h shape: (Batch, Seq_Len, 3 * Hidden_Dim)
                h = torch.cat([h, x_proj], dim=-1)

            # LSTM Forward
            # h shape out: (Batch, Seq_Len, 2 * Hidden_Dim)
            h, _ = layer(h)

            # Apply Normalization and Dropout
            h = norm(h)
            h = self.dropout(h)

        # 3. Final Regression
        out = self.head(h)

        return out
