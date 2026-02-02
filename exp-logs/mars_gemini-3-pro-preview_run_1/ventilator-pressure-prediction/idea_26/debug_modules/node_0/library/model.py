import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem (Inception-style).
    Processes input sequence with kernels [3, 5, 7], concatenates, and projects to bottleneck.
    """

    def __init__(self, input_dim, stem_dim):
        super().__init__()
        # Distribute output channels roughly equally among the 3 kernels
        c1 = stem_dim // 3
        c2 = stem_dim // 3
        c3 = stem_dim - c1 - c2

        self.conv3 = nn.Conv1d(input_dim, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, c2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, c3, kernel_size=7, padding=3)

        self.act = nn.GELU()
        self.project = nn.Linear(stem_dim, stem_dim)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        x = x.transpose(1, 2)  # (Batch, Input_Dim, Seq)

        o3 = self.conv3(x)
        o5 = self.conv5(x)
        o7 = self.conv7(x)

        # Concatenate along channel dimension
        out = torch.cat([o3, o5, o7], dim=1)
        out = self.act(out)

        out = out.transpose(1, 2)  # (Batch, Seq, Stem_Dim)
        out = self.project(out)
        return out


class CompositeBlock(nn.Module):
    """
    Wide-State Identity Block with Explicit Context Injection.
    Structure:
    1. Cat(Input, Context) -> Bi-LSTM
    2. Residual(Input, LSTM_Out)
    3. FFN
    4. Residual(Input, FFN_Out)
    """

    def __init__(self, model_dim, context_dim, hidden_dim, expansion_factor, dropout):
        super().__init__()

        # Bi-LSTM
        # Input: Model State + Context
        self.lstm = nn.LSTM(
            input_size=model_dim + context_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Ensure dimensions align for identity residual
        assert (
            hidden_dim * 2 == model_dim
        ), f"Bi-LSTM output ({hidden_dim*2}) must match model_dim ({model_dim})"

        self.dropout = nn.Dropout(dropout)

        # Pointwise FFN
        ffn_dim = model_dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, model_dim)
        )

    def forward(self, x, context):
        # x: (Batch, Seq, Model_Dim)
        # context: (Batch, Seq, Context_Dim)

        # 1. Explicit Context Injection
        lstm_input = torch.cat([x, context], dim=-1)

        # 2. Bi-LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # 3. Strict Identity Residual 1
        x = x + self.dropout(lstm_out)

        # 4. FFN + Strict Identity Residual 2
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return x


class WideProjectedNet(nn.Module):
    """
    Wide-Projected Deeply-Supervised Physics-Identity Network.
    """

    def __init__(self, input_dim, feature_names=None):
        super().__init__()

        # Config
        stem_dim = Config.STEM_DIM
        model_dim = Config.MODEL_DIM
        hidden_dim = Config.HIDDEN_DIM
        num_blocks = Config.NUM_BLOCKS
        expansion = Config.EXPANSION_FACTOR
        dropout = Config.DROPOUT
        self.aux_index = Config.AUX_BLOCK_INDEX

        # Identify Context Indices
        # We explicitly look for static and physics attributes to inject into the backbone
        self.context_indices = []
        target_context = ["R", "C", "R_u_in", "vol_C"]

        if feature_names is not None:
            for i, name in enumerate(feature_names):
                if name in target_context:
                    self.context_indices.append(i)

        context_dim = len(self.context_indices)

        # Architecture
        self.stem = MultiScaleStem(input_dim, stem_dim)

        # Adapter: Projects from compressed Stem dim to Wide Model dim
        self.adapter = nn.Linear(stem_dim, model_dim)

        self.blocks = nn.ModuleList(
            [
                CompositeBlock(model_dim, context_dim, hidden_dim, expansion, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Linear(model_dim, 1)
        self.aux_head = nn.Linear(model_dim, 1)

        self._init_weights()

    def _init_weights(self):
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
        # x: (Batch, Seq, Input_Dim)

        # Extract Context
        if self.context_indices:
            context = x[:, :, self.context_indices]
        else:
            # Handle case where no context is found (e.g. if feature_names not provided)
            context = torch.empty(x.size(0), x.size(1), 0, device=x.device)

        # Stem
        stem_out = self.stem(x)

        # Projection to Wide State
        h = self.adapter(stem_out)

        aux_pred = None

        # Backbone
        for i, block in enumerate(self.blocks):
            h = block(h, context)

            if i == self.aux_index:
                aux_pred = self.aux_head(h)

        final_pred = self.head(h)

        return final_pred, aux_pred
