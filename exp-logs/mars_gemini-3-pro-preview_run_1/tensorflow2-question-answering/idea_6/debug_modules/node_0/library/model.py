import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class IMCN(nn.Module):
    """
    Interaction Map Convolutional Network (IMCN).

    This model treats the similarity between Question and Candidate Answer tokens as an image (Interaction Map)
    and uses 2D Convolutions to detect semantic matching patterns.
    """

    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix (np.ndarray): Pre-trained embedding matrix of shape (vocab_size, embed_dim).
        """
        super(IMCN, self).__init__()

        # 1. Embedding Layer
        # Initialize with pre-trained weights and freeze them (static embeddings)
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        # 2. Feature Extractor (2D Convolution)
        # Input channel is 1 (the interaction map), output is NUM_FILTERS
        # Padding is set to maintain spatial dimensions (same padding logic)
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=Config.NUM_FILTERS,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # 3. Long Answer Head
        # Global Max Pooling reduces (Batch, Filters, Q, C) to (Batch, Filters)
        self.la_dense1 = nn.Linear(Config.NUM_FILTERS, Config.HIDDEN_DIM)
        self.la_dense2 = nn.Linear(Config.HIDDEN_DIM, 1)

        # 4. Short Answer Head
        # Column-wise Max Pooling reduces (Batch, Filters, Q, C) to (Batch, Filters, C)
        # 1D Convolution maps features to 2 channels (Start Logits, End Logits)
        self.sa_conv1d = nn.Conv1d(
            in_channels=Config.NUM_FILTERS,
            out_channels=2,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )

    def forward(self, q_indices, c_indices):
        """
        Args:
            q_indices: (Batch, Q_Len) - Token indices for Questions
            c_indices: (Batch, C_Len) - Token indices for Candidates

        Returns:
            la_logits: (Batch, 1) - Logits for Long Answer classification
            start_logits: (Batch, C_Len) - Logits for Short Answer start position
            end_logits: (Batch, C_Len) - Logits for Short Answer end position
        """
        # -------------------------------------------------------
        # 1. Generate Interaction Matrix
        # -------------------------------------------------------
        # Embed inputs: (Batch, Seq_Len, Embed_Dim)
        q_embed = self.embedding(q_indices)
        c_embed = self.embedding(c_indices)

        # Compute Similarity Matrix (Dot Product)
        # (B, Q, E) x (B, E, C) -> (B, Q, C)
        # We transpose c_embed to align dimensions for matrix multiplication
        interaction_map = torch.bmm(q_embed, c_embed.transpose(1, 2))

        # Add channel dimension for CNN: (B, 1, Q, C)
        x = interaction_map.unsqueeze(1)

        # -------------------------------------------------------
        # 2. 2D Pattern Extraction
        # -------------------------------------------------------
        # Apply 2D Conv -> ReLU -> Dropout
        # Output: (B, Num_Filters, Q, C)
        x = F.relu(self.conv2d(x))
        x = self.dropout(x)

        # -------------------------------------------------------
        # 3. Long Answer Prediction
        # -------------------------------------------------------
        # Global Max Pooling: Max over both Q (dim 2) and C (dim 3) dimensions
        # Result: (B, Num_Filters)
        la_feat = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        la_feat = la_feat.view(la_feat.size(0), -1)  # Flatten

        # Dense layers
        la_out = F.relu(self.la_dense1(la_feat))
        la_out = self.dropout(la_out)
        la_logits = self.la_dense2(la_out)  # (B, 1)

        # -------------------------------------------------------
        # 4. Short Answer Prediction
        # -------------------------------------------------------
        # Relevance Projection: Pool along Question axis (dim 2)
        # This collapses the Q dimension, keeping the C dimension (sequence length)
        # x shape: (B, Filters, Q, C) -> (B, Filters, C)
        sa_feat, _ = torch.max(x, dim=2)

        # Apply 1D Conv to get start/end scores
        # Output: (B, 2, C)
        sa_logits = self.sa_conv1d(sa_feat)

        # Split into start and end logits
        start_logits = sa_logits[:, 0, :]  # (B, C)
        end_logits = sa_logits[:, 1, :]  # (B, C)

        return la_logits, start_logits, end_logits
