import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HAPSModel(nn.Module):
    """
    Hybrid Anchor-Pairwise Sorting Network (HAPS).

    This model predicts the position of markdown cells within a notebook using two heads:
    1. Anchor Head: Predicts the coarse position relative to code cells (which act as anchors).
    2. Pairwise Head: Predicts the relative order of markdown cells that fall within the same anchor interval.
    """

    def __init__(self):
        super(HAPSModel, self).__init__()

        input_dim = Config.input_dim
        hidden_dim = Config.hidden_dim
        projection_dim = Config.projection_dim
        dropout_rate = Config.dropout

        # --- 1. Projection Towers ---
        # Project Code and Markdown embeddings into a shared latent space
        self.code_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.md_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # --- 2. Anchor Head Components ---
        # A learnable parameter representing the interval "After the last code cell".
        # Code cells C_0 ... C_N create N+1 intervals.
        # The N code cells act as keys for the first N intervals.
        # This parameter acts as the key for the (N+1)-th interval.
        self.final_bin_embedding = nn.Parameter(torch.randn(1, 1, projection_dim))

        # --- 3. Pairwise Head Components ---
        # Bilinear layer to score the relationship between two markdown cells (Mi, Mj).
        # Output > 0 implies Mi precedes Mj.
        self.pairwise_bilinear = nn.Bilinear(projection_dim, projection_dim, 1)

    def forward(
        self, code_embeddings, code_mask, md_embeddings, md_mask, pairwise_indices=None
    ):
        """
        Args:
            code_embeddings: (Batch, MaxCode, InputDim)
            code_mask: (Batch, MaxCode) - Boolean mask (True = valid)
            md_embeddings: (Batch, MaxMD, InputDim)
            md_mask: (Batch, MaxMD) - Boolean mask (True = valid)
            pairwise_indices: (NumPairs, 3) - [batch_idx, md_idx_1, md_idx_2] LongTensor

        Returns:
            dict containing:
                'anchor_logits': (Batch, MaxMD, MaxCode + 1)
                'pairwise_logits': (NumPairs,) or None
        """
        batch_size = code_embeddings.shape[0]

        # 1. Project Embeddings
        # (B, N_code, ProjDim)
        proj_code = self.code_projector(code_embeddings)
        # (B, N_md, ProjDim)
        proj_md = self.md_projector(md_embeddings)

        # 2. Anchor Head Forward Pass
        # Prepare Keys: Concatenate code embeddings with the learnable final bin embedding
        # final_bin_embedding is (1, 1, D), broadcast to (B, 1, D)
        final_bin = self.final_bin_embedding.expand(batch_size, -1, -1)

        # Keys shape: (B, N_code + 1, ProjDim)
        keys = torch.cat([proj_code, final_bin], dim=1)

        # Queries shape: (B, N_md, ProjDim)
        queries = proj_md

        # Compute Scaled Dot-Product Attention
        # (B, N_md, D) @ (B, D, N_code+1) -> (B, N_md, N_code+1)
        scale = Config.projection_dim**-0.5
        anchor_logits = torch.bmm(queries, keys.transpose(1, 2)) * scale

        # Masking
        # We need to mask out invalid code cells in the keys.
        # The final bin (index N_code) is always valid.
        # code_mask is (B, N_code). We append a column of True for the final bin.
        final_mask_col = torch.ones(
            (batch_size, 1), device=code_mask.device, dtype=torch.bool
        )
        extended_key_mask = torch.cat(
            [code_mask, final_mask_col], dim=1
        )  # (B, N_code+1)

        # Expand mask for broadcasting: (B, 1, N_code+1)
        extended_key_mask = extended_key_mask.unsqueeze(1)

        # Apply mask: set logits of invalid keys to -inf
        anchor_logits = anchor_logits.masked_fill(~extended_key_mask, float("-inf"))

        # 3. Pairwise Head Forward Pass
        pairwise_logits = None
        if pairwise_indices is not None and pairwise_indices.shape[0] > 0:
            # pairwise_indices is (K, 3): [batch_idx, md_idx_1, md_idx_2]
            b_idx = pairwise_indices[:, 0]
            idx1 = pairwise_indices[:, 1]
            idx2 = pairwise_indices[:, 2]

            # Gather embeddings using advanced indexing
            # proj_md is (B, MaxMD, D)
            vec1 = proj_md[b_idx, idx1]  # (K, D)
            vec2 = proj_md[b_idx, idx2]  # (K, D)

            # Compute bilinear score
            # Output is (K, 1), squeeze to (K,)
            pairwise_logits = self.pairwise_bilinear(vec1, vec2).squeeze(-1)

        return {"anchor_logits": anchor_logits, "pairwise_logits": pairwise_logits}
