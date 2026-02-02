import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBiLSTM(nn.Module):
    """
    Deep Residual Bidirectional LSTM with Input Injection.
    Unified architecture that outperforms split physics-based streams.

    Cite solution_lesson_node_00025: Unified Recurrent Architectures Outperform Physics-Based Branching
    Cite solution_lesson_node_00021: Input Injection in Deep Recurrent Networks
    Cite solution_lesson_node_00023: Stabilizing Deep Residual Recurrent Networks with Layer Normalization
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

        # Final projection to scalar pressure
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
            output, _ = self.lstm_layers[i](lstm_input)

            # Apply LayerNorm and Dropout
            output = self.layer_norms[i](output)
            output = self.dropouts[i](output)

            # Residual Connection
            # Layer 0 output is (B, L, 2*H). Input was (B, L, F). No residual.
            # Layer k output is (B, L, 2*H). Input from prev layer was (B, L, 2*H).
            if i > 0:
                output = output + current_features

            current_features = output

        # Project to scalar
        return self.head(current_features)
