import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SignedSinusoidalEncoding(nn.Module):
    """
    Encodes signed scalar distances using sinusoidal functions.
    Preserves the sign of the distance to distinguish upstream/downstream dependencies.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Div term for sinusoidal frequencies: 10000^(2i/d_model)
        # We compute this once and register as buffer
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, 1) containing signed distances.
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        # x is (N, L, 1), div_term is (D/2)
        # phase becomes (N, L, D/2)
        phase = x * self.div_term

        # sin preserves sign (odd function), cos is even (magnitude focused)
        pe_sin = torch.sin(phase)
        pe_cos = torch.cos(phase)

        # Concatenate to get (N, L, D)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    A single block of Pre-LayerNorm Residual BiGRU.
    Structure: x + Dropout(Projection(BiGRU(LayerNorm(x))))
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # BiGRU outputs 2 * hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # Project back to hidden_dim to allow residual connection
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # Pre-Norm
        residual = x
        out = self.norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Projection & Dropout
        out = self.proj(out)
        out = self.dropout(out)

        # Residual connection
        return residual + out


class StructureShortcutResBiGRU(nn.Module):
    """
    Main Architecture: Structure-Shortcut Deep Residual BiGRU.
    Features:
    1. Multi-modal Embedding (Seq + Loop + Geometric Distance)
    2. Deep Pre-LN ResBiGRU Backbone
    3. Structural Shortcut Head (Gathering partner states)
    4. MLP Classifier
    """

    def __init__(self):
        super().__init__()

        # --- 1. Embeddings ---
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBEDDING_DIM)
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBEDDING_DIM)
        self.dist_embed = SignedSinusoidalEncoding(Config.EMBEDDING_DIM)

        # Total input dimension
        input_dim = Config.EMBEDDING_DIM * 3

        # Project input to hidden dim if necessary, or ensure they match
        # Here Config.HIDDEN_DIM is 384, and 128*3 = 384, so they match perfectly.
        self.input_proj = nn.Identity()
        if input_dim != Config.HIDDEN_DIM:
            self.input_proj = nn.Linear(input_dim, Config.HIDDEN_DIM)

        # --- 2. Backbone ---
        self.layers = nn.ModuleList(
            [
                ResidualBiGRUBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # Final Norm after the stack (standard for Pre-LN)
        self.final_norm = nn.LayerNorm(Config.HIDDEN_DIM)

        # --- 3. Classifier / Head ---
        # Input to classifier is fused state: [h_local; h_partner] -> 2 * HIDDEN_DIM
        self.head = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM * 2, Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS),
        )

    def forward(self, sequence, loop_type, pair_index, pair_dist, **kwargs):
        """
        Args:
            sequence: (N, L) LongTensor
            loop_type: (N, L) LongTensor
            pair_index: (N, L) LongTensor (indices of paired bases)
            pair_dist: (N, L, 1) FloatTensor (signed distances)
        """
        # 1. Embed Inputs
        emb_seq = self.seq_embed(sequence)  # (N, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (N, L, 128)
        emb_dist = self.dist_embed(pair_dist)  # (N, L, 128)

        # Concatenate
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (N, L, 384)
        x = self.input_proj(x)

        # 2. Backbone Processing
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)  # (N, L, 384)

        # 3. Structural Shortcut Head
        # Gather partner states
        # pair_index is (N, L). We need to gather from dim 1 of x.
        # Create batch indices for gathering
        batch_size, seq_len, _ = x.shape

        # Flatten batch and sequence for easier gathering or use gather
        # torch.gather requires index to have same dims as input
        # We construct the gather index for the hidden dim
        # However, it's easier to use index_select logic per batch or expansion

        # Efficient gathering:
        # We want out[b, i] = x[b, pair_index[b, i]]
        # Expand pair_index to (N, L, D) is expensive.
        # Let's use the fact that pair_index contains indices in [0, L-1]

        # Create a batch offset to flatten the batch dimension
        # flat_idx = b * L + pair_index[b, i]
        batch_offset = torch.arange(batch_size, device=x.device).unsqueeze(1) * seq_len
        flat_pair_index = (batch_offset + pair_index).view(-1)  # (N*L)

        flat_x = x.view(-1, Config.HIDDEN_DIM)  # (N*L, D)
        h_partner = flat_x.index_select(0, flat_pair_index)  # (N*L, D)
        h_partner = h_partner.view(batch_size, seq_len, Config.HIDDEN_DIM)

        # Apply Mask for unpaired bases
        # Unpaired bases have pair_dist == 0.0 (and pair_index points to self in preprocessing)
        # We can derive mask from pair_dist != 0
        # pair_dist is (N, L, 1)
        mask = (pair_dist != 0).float()
        h_partner = h_partner * mask

        # Fuse: Concatenate local and partner states
        h_fused = torch.cat([x, h_partner], dim=-1)  # (N, L, 2*D)

        # 4. Classification
        logits = self.head(h_fused)  # (N, L, 3)

        return logits
