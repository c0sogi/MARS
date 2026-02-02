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


class ParallelContextExtractor(nn.Module):
    """
    Extracts context using three parallel branches (Static, Dynamic, Global)
    and fuses them into a bottleneck representation.
    """

    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()

        # Feature indices based on dataset.py generation order:
        # 0:time_step, 1:u_in, 2:R, 3:C, 4:volume, 5:R_u_in, 6:vol_C,
        # 7:u_in_diff1, 8:u_in_diff2, 9-12:u_in_lag1-4, 13:u_out

        # Static: R, C
        self.static_idx = [2, 3]

        # Dynamic: time, u_in, volume, interactions, diffs, lags, u_out
        self.dynamic_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        self.static_dim = len(self.static_idx)
        self.dynamic_dim = len(self.dynamic_idx)
        self.global_dim = input_dim

        # Internal dimension for GLU branches (set to bottleneck dim for capacity)
        glu_dim = bottleneck_dim

        self.glu_static = GLU(self.static_dim, glu_dim)
        self.glu_dynamic = GLU(self.dynamic_dim, glu_dim)
        self.glu_global = GLU(self.global_dim, glu_dim)

        # Fusion layer: Concatenation of 3 branches -> Bottleneck
        self.fusion = nn.Linear(glu_dim * 3, bottleneck_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x shape: (Batch, Seq, Features)

        x_static = x[:, :, self.static_idx]
        x_dynamic = x[:, :, self.dynamic_idx]

        out_static = self.glu_static(x_static)
        out_dynamic = self.glu_dynamic(x_dynamic)
        out_global = self.glu_global(x)

        # Concatenate branches
        fused = torch.cat([out_static, out_dynamic, out_global], dim=-1)

        # Project to bottleneck ("Mild Bottleneck")
        context = self.fusion(fused)
        context = self.activation(context)

        return context


class PCGIBiLSTM(nn.Module):
    """
    Parallel-Context Gated-Injection BiLSTM (PC-GI-BiLSTM).
    Features:
    - Parallel Context Extraction (Static/Dynamic/Global separation).
    - Dual-Path Injection Payload (Context + Raw Identity).
    - Deep Injection (Payload fed to every LSTM layer).
    - Stretched Deep Backbone (4 layers).
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        # Continuous features (13) + u_out (1) = 14
        self.input_dim = 14
        self.hidden_dim = Config.LSTM_HIDDEN_DIM
        self.bottleneck_dim = Config.BOTTLENECK_DIM
        self.num_layers = Config.LSTM_LAYERS

        # 1. Context Extractor
        self.context_extractor = ParallelContextExtractor(
            self.input_dim, self.bottleneck_dim
        )

        # 2. Payload Dimension
        # Payload = Context (Bottleneck) + Identity (Raw Input)
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
