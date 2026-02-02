import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Inception-style stem that captures multi-scale local features (kernels 3, 5, 7)
    and projects them to a compressed bottleneck dimension.
    """

    def __init__(self, input_dim, stem_dim):
        super().__init__()
        # Use a consistent number of filters for intermediate branches
        inter_dim = stem_dim

        # Multi-scale convolutions
        self.conv3 = nn.Conv1d(input_dim, inter_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, inter_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, inter_dim, kernel_size=7, padding=3)

        # Projection to bottleneck dimension (Stem Dim)
        self.project = nn.Conv1d(inter_dim * 3, stem_dim, kernel_size=1)

        self.act = nn.GELU()

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq) for Conv1d
        x = x.transpose(1, 2)

        # Apply branches
        c3 = self.act(self.conv3(x))
        c5 = self.act(self.conv5(x))
        c7 = self.act(self.conv7(x))

        # Concatenate along channel dimension
        out = torch.cat([c3, c5, c7], dim=1)

        # Project to target stem dimension
        out = self.project(out)
        out = self.act(out)

        # Return to (Batch, Seq, Dim)
        return out.transpose(1, 2)


class CompositeBlock(nn.Module):
    """
    Wide-State Identity Block.
    Injects physics context at every step, applies Bi-LSTM mixing, and uses
    strict identity residuals without LayerNorm to preserve pressure magnitude.
    """

    def __init__(self, model_dim, input_dim, lstm_hidden, dropout, expansion=2):
        super().__init__()

        # 1. Wide-State Temporal Mixing (Bi-LSTM)
        # Input: Latent State (model_dim) + Physics Context (input_dim)
        # Output: 2 * lstm_hidden. We expect 2*lstm_hidden == model_dim.
        self.lstm = nn.LSTM(
            input_size=model_dim + input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        # 2. Pointwise Channel Mixing (FFN)
        ffn_dim = int(model_dim * expansion)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, model_dim)
        )

    def forward(self, h, x_context):
        """
        Args:
            h: Latent state tensor of shape (Batch, Seq, Model_Dim)
            x_context: Raw/Engineered feature tensor of shape (Batch, Seq, Input_Dim)
        """
        # Curated Physics Injection: Concatenate context to latent state
        lstm_input = torch.cat([h, x_context], dim=-1)

        # LSTM Mixing
        lstm_out, _ = self.lstm(lstm_input)

        # Strict Identity Residual 1 (No weights, no norm)
        h = h + self.dropout(lstm_out)

        # FFN Mixing
        ffn_out = self.ffn(h)

        # Strict Identity Residual 2 (No weights, no norm)
        h = h + self.dropout(ffn_out)

        return h


class WideProjectedNet(nn.Module):
    """
    Wide-Projected Physics-Composite Network.
    Implements a Bottleneck-to-Wide topology with deep supervision.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.stem_dim = Config.STEM_DIM
        self.model_dim = Config.MODEL_DIM
        self.lstm_hidden = Config.LSTM_HIDDEN
        self.dropout_rate = Config.DROPOUT
        self.aux_idx = Config.AUX_LAYER_INDEX

        # 1. Stem (Bottleneck Initialization)
        self.stem = MultiScaleStem(input_dim, self.stem_dim)

        # 2. Wide Projection Adapter
        # Projects compressed stem features to high-capacity model dimension
        self.projection = nn.Linear(self.stem_dim, self.model_dim)

        # 3. Backbone (Composite Blocks)
        self.layers = nn.ModuleList(
            [
                CompositeBlock(
                    model_dim=self.model_dim,
                    input_dim=input_dim,
                    lstm_hidden=self.lstm_hidden,
                    dropout=self.dropout_rate,
                    expansion=Config.FFN_EXPANSION,
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Heads
        self.aux_head = nn.Linear(self.model_dim, 1)
        self.head = nn.Linear(self.model_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq, Input_Dim)
        Returns:
            final_out: Prediction from the last block (Batch, Seq, 1)
            aux_out: Prediction from the auxiliary block (Batch, Seq, 1)
        """
        # Stem
        stem_out = self.stem(x)

        # Projection to Wide State
        h = self.projection(stem_out)

        aux_out = None

        # Backbone
        for i, layer in enumerate(self.layers):
            # Pass both latent state and physics context
            h = layer(h, x)

            # Deep Supervision
            if i == self.aux_idx:
                aux_out = self.aux_head(h)

        # Final Prediction
        final_out = self.head(h)

        return final_out, aux_out
