import torch
import torch.nn as nn
from library.config import Config


class GLU(nn.Module):
    """
    Gated Linear Unit for Monolithic Context Extraction.
    Projects input to 2 * output_dim, splits into value and gate,
    and applies sigmoid activation to the gate.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        out, gate = out.chunk(2, dim=-1)
        return out * torch.sigmoid(gate)


class WCMI_BiLSTM(nn.Module):
    """
    Wide-Context Monolithic-Injection BiLSTM (WCMI-BiLSTM).

    Architecture:
    1. Monolithic Context Extractor (GLU): Processes all inputs to capture interactions.
    2. Dual-Path Injection: Concatenates GLU context with raw input (Identity Path).
    3. Deep Recurrent Backbone: 4-layer BiLSTM with Deep Injection (input fed to all layers).
    """

    def __init__(self, input_dim):
        super().__init__()
        self.hidden_size = Config.HIDDEN_SIZE

        # 1. Wide Monolithic Context Extractor
        self.glu = GLU(input_dim, Config.GLU_DIM)

        # 2. Injection Payload Dimension: GLU output + Raw Input (Identity Path)
        self.injection_dim = Config.GLU_DIM + input_dim

        # 3. Deep Recurrent Backbone
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList()

        for i in range(Config.NUM_LAYERS):
            # Deep Injection Logic:
            # Layer 0: Input is just the injection payload.
            # Layer >0: Input is concatenation of Previous Layer Output + Injection Payload.
            if i == 0:
                layer_input_dim = self.injection_dim
            else:
                # BiLSTM output is hidden_size * 2
                layer_input_dim = (self.hidden_size * 2) + self.injection_dim

            self.layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.lns.append(nn.LayerNorm(self.hidden_size * 2))

        self.dropout = nn.Dropout(Config.DROPOUT)

        # 4. Prediction Head
        self.head = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        # x shape: [Batch, Seq, Feat]

        # 1. Context Extraction
        context = self.glu(x)

        # 2. Construct Injection Payload (Context + Identity)
        injection = torch.cat([context, x], dim=-1)

        curr_input = injection

        # 3. Deep Backbone with Deep Injection
        for i, lstm in enumerate(self.layers):
            if i > 0:
                # Concatenate injection payload to previous layer output for deep layers
                curr_input = torch.cat([curr_input, injection], dim=-1)

            output, _ = lstm(curr_input)

            # Apply LayerNorm and Dropout between layers
            output = self.lns[i](output)
            output = self.dropout(output)

            curr_input = output

        # 4. Head
        # Project to scalar pressure
        pred = self.head(curr_input).squeeze(-1)
        return pred


# Alias to match potential naming variations in description
WCMIBiLSTM = WCMI_BiLSTM
