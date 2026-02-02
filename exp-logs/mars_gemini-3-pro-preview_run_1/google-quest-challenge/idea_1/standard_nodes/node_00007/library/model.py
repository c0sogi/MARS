import torch
import torch.nn as nn
from library.config import Config


class DualBranchDAN(nn.Module):
    """
    Dual-Branch Deep Averaging Network (DAN) for Question-Answer pair classification.

    This model processes the Question and Answer text independently using a shared
    embedding layer and global average pooling. The resulting representations are
    concatenated and passed through an MLP to predict multiple target probabilities.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        num_targets=len(Config.TARGET_COLS),
    ):
        """
        Initializes the DualBranchDAN model.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to Config.VOCAB_SIZE.
            embedding_dim (int): Dimension of the embedding vectors. Defaults to Config.EMBEDDING_DIM.
            hidden_dim (int): Dimension of the hidden layer in the MLP. Defaults to Config.HIDDEN_DIM.
            dropout (float): Dropout probability. Defaults to Config.DROPOUT.
            num_targets (int): Number of target labels to predict. Defaults to len(Config.TARGET_COLS).
        """
        super(DualBranchDAN, self).__init__()

        # Shared Embedding Layer
        # padding_idx=0 ensures that the padding token (index 0) maps to a zero vector
        # and does not accumulate gradients.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        # MLP Head
        # Input: [u, v, u*v, |u-v|] where u, v are (Avg + Max) pooled vectors.
        # u dim = 2 * embedding_dim
        # v dim = 2 * embedding_dim
        # Interaction terms match u dim.
        # Total input dim = 4 * (2 * embedding_dim) = 8 * embedding_dim
        self.fc1 = nn.Linear(embedding_dim * 8, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_targets)
        self.sigmoid = nn.Sigmoid()

    def _pool(self, embed, mask):
        """
        Applies Masked Global Average Pooling and Masked Global Max Pooling.
        Cite solution_lesson_node_00003
        """
        # embed: (B, L, D)
        # mask: (B, L) - 1 for valid, 0 for pad

        # 1. Masked Average Pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_embed = torch.sum(embed * mask_expanded, dim=1)
        lengths = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        avg_pool = sum_embed / lengths

        # 2. Masked Max Pooling
        # Replace pad positions with very small number before max
        is_pad = (mask == 0).unsqueeze(-1)
        embed_for_max = embed.clone()
        embed_for_max = embed_for_max.masked_fill(is_pad, -1e9)
        max_pool, _ = torch.max(embed_for_max, dim=1)

        # Handle case where sequence is all padding (max would be -1e9)
        is_empty = (mask.sum(dim=1, keepdim=True) == 0).float()
        max_pool = max_pool * (1 - is_empty)

        return torch.cat([avg_pool, max_pool], dim=1)

    def forward(self, q_seq, a_seq):
        """
        Forward pass with Dual-Pooling and Interaction Heuristics.
        """
        # Create masks (assuming 0 is pad_idx)
        q_mask = (q_seq != 0).long()
        a_mask = (a_seq != 0).long()

        # Embed
        q_emb = self.embedding(q_seq)
        a_emb = self.embedding(a_seq)

        # Pool (Cite solution_lesson_node_00003)
        # u, v shape: (B, 2 * embedding_dim)
        u = self._pool(q_emb, q_mask)
        v = self._pool(a_emb, a_mask)

        # Interaction Heuristics (Cite solution_lesson_node_00004)
        # Element-wise product and Absolute difference
        prod = u * v
        diff = torch.abs(u - v)

        # Concatenate: [u, v, u*v, |u-v|]
        combined = torch.cat([u, v, prod, diff], dim=1)

        # MLP Head
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        out = self.sigmoid(x)

        return out
