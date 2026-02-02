import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class RecurrentStream(nn.Module):
    """
    Stream 1: The Recurrent "Elastic" Path.
    Models time-dependent dynamics (Volume, Compliance) using a Deep Residual Bidirectional LSTM
    with Input Injection.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.LSTM_LAYERS
        self.dropout_p = Config.LSTM_DROPOUT
        self.bidirectional = Config.LSTM_BIDIRECTIONAL

        # Calculate output dimension of LSTM layers
        self.lstm_out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Input Injection:
            # Layer 0 takes raw input.
            # Layer k > 0 takes concat(prev_layer_output, raw_input).
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = self.lstm_out_dim + input_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=self.hidden_dim,
                    bidirectional=self.bidirectional,
                    batch_first=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(self.lstm_out_dim))
            self.dropouts.append(nn.Dropout(self.dropout_p))

        # Final projection to scalar pressure component
        self.head = nn.Linear(self.lstm_out_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        current_features = x

        for i in range(self.num_layers):
            # Input Injection for layers > 0
            if i > 0:
                # Concatenate original input x to the features from previous layer
                lstm_input = torch.cat([current_features, x], dim=-1)
            else:
                lstm_input = current_features

            # Pass through LSTM
            # output shape: (Batch, Seq_Len, Direction * Hidden)
            output, _ = self.lstm_layers[i](lstm_input)

            # Apply LayerNorm and Dropout
            output = self.layer_norms[i](output)
            output = self.dropouts[i](output)

            # Residual Connection
            # We can only add residual if shapes match.
            # Layer 0 output is (B, L, 2*H). Input was (B, L, F). No residual.
            # Layer k output is (B, L, 2*H). Input from prev layer was (B, L, 2*H).
            # So for i > 0, current_features holds the output of layer i-1.
            if i > 0:
                output = output + current_features

            current_features = output

        # Project to scalar
        # Shape: (Batch, Seq_Len, 1)
        elastic_pressure = self.head(current_features)
        return elastic_pressure


class InstantaneousStream(nn.Module):
    """
    Stream 2: The Instantaneous "Resistive" Path.
    Models flow-dependent dynamics (Resistance, Flow) using a Deep Residual MLP
    with Input Injection. Operates on each time step independently.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.hidden_dim = Config.MLP_DIM
        self.num_layers = Config.MLP_LAYERS
        self.dropout_p = Config.MLP_DROPOUT

        self.linear_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Input Injection:
            # Layer 0 takes raw input.
            # Layer k > 0 takes concat(prev_layer_output, raw_input).
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = self.hidden_dim + input_dim

            self.linear_layers.append(nn.Linear(layer_input_dim, self.hidden_dim))
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout_p))

        # Final projection to scalar pressure component
        self.head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        current_features = x

        for i in range(self.num_layers):
            # Input Injection
            if i > 0:
                mlp_input = torch.cat([current_features, x], dim=-1)
            else:
                mlp_input = current_features

            # Linear -> GELU -> Norm -> Dropout
            output = self.linear_layers[i](mlp_input)
            output = F.gelu(output)
            output = self.layer_norms[i](output)
            output = self.dropouts[i](output)

            # Residual Connection
            if i > 0:
                output = output + current_features

            current_features = output

        # Project to scalar
        # Shape: (Batch, Seq_Len, 1)
        resistive_pressure = self.head(current_features)
        return resistive_pressure


class DSPIN(nn.Module):
    """
    Dual-Stream Physics-Informed Network.
    Combines Recurrent and Instantaneous streams to predict Airway Pressure.
    P_total = P_elastic (Stream 1) + P_resistive (Stream 2)
    """

    def __init__(self, input_dim):
        super().__init__()
        self.recurrent_stream = RecurrentStream(input_dim)
        self.instantaneous_stream = InstantaneousStream(input_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            pressure: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # Stream 1: Elastic Component (Time-dependent)
        p_elastic = self.recurrent_stream(x)

        # Stream 2: Resistive Component (Instantaneous)
        p_resistive = self.instantaneous_stream(x)

        # Physics-Informed Fusion: Additive
        p_total = p_elastic + p_resistive

        return p_total
