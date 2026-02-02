import torch
import torch.nn as nn
from library.config import Config


class GLU(nn.Module):
    """
    Gated Linear Unit:
    GLU(x) = (x * W + b) [:, :d] * sigmoid((x * W + b) [:, d:])
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class DPGIBiLSTM(nn.Module):
    """
    Dual-Path Gated-Injection BiLSTM (DP-GI-BiLSTM).
    Optimized based on Lesson ID: solution_lesson_node_00059.
    Features:
    - Monolithic GLU Context Extraction (No partitioning).
    - Dual-Path Injection Payload (Context + Raw Identity).
    - Deep Injection (Payload fed to every LSTM layer).
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        # Continuous features (13) + u_out (1) = 14
        self.input_dim = 14
        self.hidden_dim = Config.LSTM_HIDDEN_DIM
        self.bottleneck_dim = Config.BOTTLENECK_DIM
        self.num_layers = Config.LSTM_LAYERS

        # 1. Context Extractor (Monolithic GLU)
        # Cite solution_lesson_node_00059: Unified projection allows learning cross-terms.
        self.context_extractor = GLU(self.input_dim, self.bottleneck_dim)

        # 2. Payload Dimension
        # Payload = Context (Bottleneck) + Identity (Raw Input)
        # Cite solution_lesson_node_00039: Signal Preservation via concatenation.
        self.payload_dim = self.bottleneck_dim + self.input_dim

        # 3. Deep Recurrent Backbone
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Input size calculation for Deep Injection:
            # Layer 0: Receives just the Payload.
            # Layer i>0: Receives [Previous Layer Output (Hidden*2) + Payload].
            if i == 0:
                input_size = self.payload_dim
            else:
                input_size = (self.hidden_dim * 2) + self.payload_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size,
                    self.hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim * 2))
            self.dropouts.append(nn.Dropout(Config.DROPOUT))

        # 4. Head
        self.head = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, x):
        # x: (Batch, Seq, 14)

        # --- Context Generation ---
        context = self.context_extractor(x)  # (B, S, Bottleneck)

        # --- Dual-Path Injection Payload ---
        # Concatenate Context with Raw Input (Identity Path)
        # No Dropout applied here to maintain signal stability
        payload = torch.cat([context, x], dim=-1)  # (B, S, Payload_Dim)

        # --- Deep Recurrent Backbone ---
        curr_input = payload
        prev_out = None

        for i in range(self.num_layers):
            if i > 0:
                # Deep Injection: Concatenate previous output with payload
                curr_input = torch.cat([prev_out, payload], dim=-1)

            # LSTM Forward
            out, _ = self.lstm_layers[i](curr_input)

            # Inter-layer Connectivity (Norm -> Dropout)
            out = self.layer_norms[i](out)
            out = self.dropouts[i](out)

            prev_out = out

        # --- Head ---
        # Project final LSTM output to pressure
        pred = self.head(prev_out)

        return pred.squeeze(-1)
