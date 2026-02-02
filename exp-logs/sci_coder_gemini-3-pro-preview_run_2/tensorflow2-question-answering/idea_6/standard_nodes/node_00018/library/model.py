import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class InteractionGridCNN(nn.Module):
    """
    A 2D Interaction-Grid Convolutional Network for Question Answering.
    Treats the Q-Context interaction as an image recognition problem.
    """

    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix (np.ndarray): Pre-trained embedding matrix of shape (vocab_size, embed_dim).
        """
        super(InteractionGridCNN, self).__init__()

        # --- 1. Embedding Layer ---
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Load weights and freeze
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        # --- 2. 2D Encoder (CNN) ---
        # Input: (Batch, 1, N, M) -> Output: (Batch, Filters, H_out, W_out)
        # We use padding=1 with kernel_size=3 to maintain spatial dimensions before pooling
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            padding=1,
        )
        self.pool2d = nn.MaxPool2d(kernel_size=Config.CNN_POOL_SIZE)

        # Calculate flattened dimension for dense layers
        # Assuming padding maintains dim, then pooling divides by pool_size
        # H_out = MAX_Q_LEN // pool_h
        # W_out = MAX_C_LEN // pool_w
        h_out = Config.MAX_Q_LEN // Config.CNN_POOL_SIZE[0]
        w_out = Config.MAX_C_LEN // Config.CNN_POOL_SIZE[1]
        self.flat_dim = Config.CNN_FILTERS * h_out * w_out

        # --- 3. Heads ---

        # Ranking Head (Binary Classification)
        self.rank_fc1 = nn.Linear(self.flat_dim, Config.HIDDEN_DIM)
        self.rank_dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.rank_fc2 = nn.Linear(Config.HIDDEN_DIM, 1)

        # Yes/No Head (Multi-class Classification)
        self.yn_fc1 = nn.Linear(self.flat_dim, Config.HIDDEN_DIM)
        self.yn_dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.yn_fc2 = nn.Linear(Config.HIDDEN_DIM, Config.NUM_YN_CLASSES)

        # Span Prediction Head (1D CNN)
        # Input: (Batch, 1, M) derived from collapsing grid
        self.span_conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=Config.SPAN_CNN_FILTERS,
            kernel_size=Config.SPAN_CNN_KERNEL_SIZE,
            padding=Config.SPAN_CNN_KERNEL_SIZE // 2,
        )
        self.span_conv2 = nn.Conv1d(
            in_channels=Config.SPAN_CNN_FILTERS,
            out_channels=2,  # Output channel 0 for Start, 1 for End
            kernel_size=Config.SPAN_CNN_KERNEL_SIZE,
            padding=Config.SPAN_CNN_KERNEL_SIZE // 2,
        )

    def forward(self, q_ids, c_ids):
        """
        Args:
            q_ids (torch.Tensor): Question token IDs (Batch, N)
            c_ids (torch.Tensor): Candidate token IDs (Batch, M)

        Returns:
            dict: Contains 'rank_logits', 'start_logits', 'end_logits', 'yn_logits'
        """
        # 1. Embeddings
        q_emb = self.embedding(q_ids)  # (B, N, D)
        c_emb = self.embedding(c_ids)  # (B, M, D)

        # 2. Compute Interaction Grid
        # Dot product: (B, N, D) x (B, D, M) -> (B, N, M)
        interaction_grid = torch.matmul(q_emb, c_emb.transpose(1, 2))

        # Add channel dimension for CNN: (B, 1, N, M)
        x = interaction_grid.unsqueeze(1)

        # 3. 2D Encoder Flow
        x_enc = F.relu(self.conv2d(x))
        x_enc = self.pool2d(x_enc)
        x_flat = x_enc.view(x_enc.size(0), -1)

        # 4. Ranking Head
        rank_out = F.relu(self.rank_fc1(x_flat))
        rank_out = self.rank_dropout(rank_out)
        rank_logits = self.rank_fc2(rank_out).squeeze(-1)  # (B,)

        # 5. Yes/No Head
        yn_out = F.relu(self.yn_fc1(x_flat))
        yn_out = self.yn_dropout(yn_out)
        yn_logits = self.yn_fc2(yn_out)  # (B, 3)

        # 6. Span Head
        # Collapse grid along question axis (dim 2) -> (B, 1, M)
        # Summing similarity scores creates a "relevance profile" for the candidate text
        relevance_profile = torch.sum(x, dim=2)

        s = F.relu(self.span_conv1(relevance_profile))
        span_logits = self.span_conv2(s)  # (B, 2, M)

        start_logits = span_logits[:, 0, :]  # (B, M)
        end_logits = span_logits[:, 1, :]  # (B, M)

        return {
            "rank_logits": rank_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yn_logits": yn_logits,
        }
