import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from library.config import Config
from library.utils import create_embedding_matrix


class DilatedResidualBlock(nn.Module):
    """
    A 1D Dilated Convolutional Block with Residual Connection and Layer Normalization.

    Structure:
    Input -> Conv1d (Dilated) -> ReLU -> Dropout -> Residual Add -> LayerNorm -> Output
    """

    def __init__(
        self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float
    ):
        super(DilatedResidualBlock, self).__init__()

        # Calculate padding to maintain sequence length (same padding)
        # For sequence length L, kernel k, dilation d, padding p, stride 1:
        # L_out = (L + 2*p - d*(k-1) - 1) + 1
        # To have L_out == L, we need 2*p = d*(k-1)
        # p = d * (k - 1) / 2
        # Config.KERNEL_SIZE is 3, so p = d * 1 = d
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Hidden_Dim)
        Returns:
            Output tensor of shape (Batch, Seq_Len, Hidden_Dim)
        """
        residual = x

        # Conv1d expects (Batch, Channels, Seq_Len)
        out = x.transpose(1, 2)
        out = self.conv(out)
        out = out.transpose(1, 2)

        out = F.relu(out)
        out = self.dropout(out)

        # Residual connection
        out = out + residual

        # Layer Normalization
        out = self.norm(out)

        return out


class SingleStreamNetwork(nn.Module):
    """
    Single-Stream Dilated Residual Network for Natural Questions.

    Architecture:
    1. Embedding Layer (Frozen, initialized from cache)
    2. Projection Layer (maps embedding dim to hidden dim)
    3. Stack of DilatedResidualBlocks (exponential dilation)
    4. Multi-Head Output:
        - Ranking Head: Global Max Pooling -> MLP -> Logit
        - Span Heads: Token-wise Projection -> Start/End Logits
        - Yes/No Head: Global Max Pooling -> MLP -> Class Logits
    """

    def __init__(self, vocab: Dict[str, int]):
        super(SingleStreamNetwork, self).__init__()

        self.vocab = vocab
        self.pad_id = vocab.get(Config.PAD_TOKEN, 0)

        # --- 1. Embedding Layer ---
        # Load embedding matrix from cache (computed via library.utils)
        # We set load_cached_data=True as per requirements for deterministic processing
        emb_matrix_np = create_embedding_matrix(vocab, load_cached_data=True)
        emb_tensor = torch.tensor(emb_matrix_np, dtype=torch.float32)

        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor, freeze=True, padding_idx=self.pad_id
        )

        # --- 2. Projection Layer ---
        if Config.EMBEDDING_DIM != Config.HIDDEN_DIM:
            self.projection = nn.Linear(Config.EMBEDDING_DIM, Config.HIDDEN_DIM)
        else:
            self.projection = nn.Identity()

        # --- 3. Encoder Blocks ---
        self.encoder_blocks = nn.ModuleList()
        for dilation in Config.DILATION_RATES:
            block = DilatedResidualBlock(
                hidden_dim=Config.HIDDEN_DIM,
                kernel_size=Config.KERNEL_SIZE,
                dilation=dilation,
                dropout=Config.DROPOUT_RATE,
            )
            self.encoder_blocks.append(block)

        # --- 4. Ranking Head (Long Answer) ---
        self.ranking_head = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, 1),
        )

        # --- 5. Span Prediction Heads (Short Answer) ---
        self.start_head = nn.Linear(Config.HIDDEN_DIM, 1)
        self.end_head = nn.Linear(Config.HIDDEN_DIM, 1)

        # --- 6. Yes/No Head ---
        self.yes_no_head = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, Config.NUM_YES_NO_CLASSES),
        )

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            input_ids: Tensor of shape (Batch, Seq_Len) containing token indices.

        Returns:
            Dictionary containing output logits:
                - ranking_logits: (Batch,)
                - start_logits: (Batch, Seq_Len)
                - end_logits: (Batch, Seq_Len)
                - yes_no_logits: (Batch, Num_Classes)
        """
        # Create padding mask (Batch, Seq_Len)
        # 1.0 for valid tokens, 0.0 for padding
        mask = (input_ids != self.pad_id).float()
        mask_expanded = mask.unsqueeze(-1)  # (Batch, Seq_Len, 1)

        # Embed and Project
        x = self.embedding(input_ids)  # (B, S, Emb_Dim)
        x = self.projection(x)  # (B, S, Hidden_Dim)

        # Apply mask to initial embeddings
        x = x * mask_expanded

        # Pass through dilated blocks
        for block in self.encoder_blocks:
            x = block(x)
            # Re-apply mask to ensure padding positions remain zero
            # This is crucial for Global Max Pooling to work correctly with ReLU activations
            x = x * mask_expanded

        # Global Max Pooling for sequence-level tasks
        # Since we use ReLU in blocks, activations are >= 0. Padded areas are 0.
        # Max pooling will pick up features from valid tokens.
        pooled_features, _ = torch.max(x, dim=1)  # (B, Hidden_Dim)

        # Ranking Output
        ranking_logits = self.ranking_head(pooled_features).squeeze(-1)  # (B,)

        # Yes/No Output
        yes_no_logits = self.yes_no_head(pooled_features)  # (B, Num_Classes)

        # Span Outputs
        # Token-wise projection
        start_logits = self.start_head(x).squeeze(-1)  # (B, S)
        end_logits = self.end_head(x).squeeze(-1)  # (B, S)

        # Optional: Mask span logits for padded positions to a large negative value
        # This ensures softmax doesn't pick padding tokens during inference.
        # However, for training with CrossEntropyLoss(ignore_index=0), raw logits are fine.
        # We return raw logits here.

        return {
            "ranking_logits": ranking_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yes_no_logits": yes_no_logits,
        }
