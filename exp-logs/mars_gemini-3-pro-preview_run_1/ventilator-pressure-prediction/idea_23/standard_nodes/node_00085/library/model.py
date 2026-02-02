import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem.
    Processes input with multiple kernel sizes and projects to a compressed bottleneck dimension.
    """

    def __init__(self, input_dim, bottleneck_dim, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, input_dim, k, padding=k // 2) for k in kernel_sizes]
        )
        # Output dim is input_dim * number of kernels (concatenation)
        # Project to bottleneck
        self.proj = nn.Linear(input_dim * len(kernel_sizes), bottleneck_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (Batch, Length, Features)
        # Transpose for Conv1d: (Batch, Features, Length)
        x_in = x.transpose(1, 2)

        outs = []
        for conv in self.convs:
            outs.append(conv(x_in))

        # Concatenate along feature dimension: (Batch, Features * K, Length)
        x_cat = torch.cat(outs, dim=1)

        # Transpose back: (Batch, Length, Features * K)
        x_cat = x_cat.transpose(1, 2)

        # Project and Activate
        x_out = self.proj(x_cat)
        return self.act(x_out)


class ExpansionBlock(nn.Module):
    """
    Block 1: Expands from Bottleneck Dimension to Wide Dimension.
    Uses a projected residual connection.
    """

    def __init__(self, input_dim, hidden_dim, context_dim, dropout=0.0):
        super().__init__()
        # LSTM input receives the previous state concatenated with context
        self.lstm = nn.LSTM(
            input_size=input_dim + context_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional
            batch_first=True,
            bidirectional=True,
        )

        # Residual Projection: Input (Bottleneck) -> Hidden (Wide)
        self.res_proj = nn.Linear(input_dim, hidden_dim)

        # Pointwise FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: (B, L, input_dim)
        # context: (B, L, context_dim)

        # 1. LSTM Path
        lstm_in = torch.cat([x, context], dim=-1)
        lstm_out, _ = self.lstm(lstm_in)

        # 2. Projected Residual
        res = self.res_proj(x)
        x = res + self.dropout(lstm_out)

        # 3. FFN Path (Identity Residual)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return x


class WideStateBlock(nn.Module):
    """
    Blocks 2-4: Maintains the Wide Dimension.
    Uses Strict Identity Residuals (no weights on skip connection).
    """

    def __init__(self, hidden_dim, context_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim + context_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: (B, L, hidden_dim)

        # 1. LSTM Path
        lstm_in = torch.cat([x, context], dim=-1)
        lstm_out, _ = self.lstm(lstm_in)

        # Strict Identity Residual
        x = x + self.dropout(lstm_out)

        # 2. FFN Path
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return x


class GraduatedCapacityNetwork(nn.Module):
    """
    Graduated-Capacity Physics-Context Composite Network.
    Features:
    - Multi-Scale Stem (Bottleneck Init)
    - Curated Physics Context Injection
    - Expansion Block (Bottleneck -> Wide)
    - Wide State Identity Blocks
    - Deep Supervision (Aux Head)
    """

    def __init__(self, input_dim):
        super().__init__()

        # Config
        self.stem_kernels = Config.STEM_KERNEL_SIZES
        self.bottleneck_dim = Config.BOTTLENECK_DIM
        self.wide_dim = Config.WIDE_DIM
        self.dropout = Config.DROPOUT

        # Context Indices based on features.py:
        # 0: R, 1: C, 6: R_u_in, 7: vol_C
        self.context_indices = [0, 1, 6, 7]
        self.context_dim = len(self.context_indices)

        # Stem
        self.stem = MultiScaleStem(input_dim, self.bottleneck_dim, self.stem_kernels)

        # Block 1 (Expansion)
        self.block1 = ExpansionBlock(
            self.bottleneck_dim, self.wide_dim, self.context_dim, self.dropout
        )

        # Blocks 2-4 (Wide Identity)
        # Config.NUM_BLOCKS is 4. Block 1 is separate. We need 3 more.
        self.blocks = nn.ModuleList(
            [
                WideStateBlock(self.wide_dim, self.context_dim, self.dropout)
                for _ in range(Config.NUM_BLOCKS - 1)
            ]
        )

        # Heads
        self.head = nn.Linear(self.wide_dim, 1)

        if Config.USE_AUX_HEAD:
            self.aux_head = nn.Linear(self.wide_dim, 1)
            # Aux head attached after Block 2.
            # self.blocks[0] corresponds to Block 2.
            self.aux_block_idx = Config.AUX_HEAD_BLOCK_IDX - 1

        # Initialize weights for stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Explicit initialization to ensure stable signal propagation.
        Linear/Conv: Kaiming Normal (Cite solution_lesson_node_00062)
        LSTM: Orthogonal
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
                    # Initialize forget gate bias to 1 for long-term memory
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)

    def forward(self, x):
        # x: (Batch, Length, Features)

        # Extract Curated Physics Context
        context = x[:, :, self.context_indices]

        # Stem
        x_stem = self.stem(x)

        # Block 1 (Expansion)
        x_curr = self.block1(x_stem, context)

        aux_out = None

        # Blocks 2-4
        for i, block in enumerate(self.blocks):
            x_curr = block(x_curr, context)

            # Aux Head Attachment
            if Config.USE_AUX_HEAD and i == self.aux_block_idx:
                aux_out = self.aux_head(x_curr)

        # Final Head
        final_out = self.head(x_curr)

        # Squeeze to (Batch, Length)
        final_out = final_out.squeeze(-1)
        if aux_out is not None:
            aux_out = aux_out.squeeze(-1)

        return final_out, aux_out
