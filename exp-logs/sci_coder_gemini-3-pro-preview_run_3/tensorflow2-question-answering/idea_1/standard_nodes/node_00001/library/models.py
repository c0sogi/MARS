import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SiameseDANRanker(nn.Module):
    """
    Siamese Deep Averaging Network (DAN) for ranking Long Answer candidates.
    Encodes Question and Candidate independently and computes a similarity score.
    """

    def __init__(self, vocab_size, padding_idx=0):
        super(SiameseDANRanker, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )

        # MLP Encoder
        # Layer 1
        self.fc1 = nn.Linear(Config.EMBEDDING_DIM, Config.RANKER_HIDDEN_DIM)
        self.dropout = nn.Dropout(Config.RANKER_DROPOUT)
        self.activation = nn.ReLU()

        # Layer 2 (Output projection)
        # We project to a vector size suitable for dot product.
        # Keeping it at hidden_dim allows rich interaction.
        self.fc2 = nn.Linear(Config.RANKER_HIDDEN_DIM, Config.RANKER_HIDDEN_DIM)

        self.padding_idx = padding_idx

    def encode(self, input_ids):
        """
        Encodes a sequence of token IDs into a dense vector.
        Args:
            input_ids: Tensor of shape (Batch, Seq_Len)
        Returns:
            Tensor of shape (Batch, Hidden_Dim)
        """
        # Mask for padding (Batch, Seq_Len)
        mask = (input_ids != self.padding_idx).float().unsqueeze(-1)  # (B, L, 1)

        # Embeddings (Batch, Seq_Len, Emb_Dim)
        embeds = self.embedding(input_ids)

        # Average Pooling (ignoring padding)
        # Sum embeddings
        sum_embeds = torch.sum(embeds * mask, dim=1)  # (B, E)
        # Sum mask (lengths)
        lengths = torch.sum(mask, dim=1)  # (B, 1)
        # Avoid division by zero
        lengths = torch.clamp(lengths, min=1e-9)

        avg_embeds = sum_embeds / lengths

        # MLP
        out = self.fc1(avg_embeds)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def forward(self, q_ids, ctx_ids):
        """
        Computes similarity scores between questions and contexts.

        Args:
            q_ids: (Batch, Q_Len)
            ctx_ids: (Batch, Ctx_Len) OR (Batch, Num_Candidates, Ctx_Len)

        Returns:
            scores: (Batch, ) or (Batch, Num_Candidates) depending on input
        """
        # Encode Question
        q_vec = self.encode(q_ids)  # (B, H)

        # Handle Context Input
        if ctx_ids.dim() == 2:
            # Case: Single context per question (e.g., positive sample)
            c_vec = self.encode(ctx_ids)  # (B, H)
            # Dot Product
            scores = torch.sum(q_vec * c_vec, dim=-1)  # (B,)

        elif ctx_ids.dim() == 3:
            # Case: Multiple candidates per question (e.g., negatives or inference)
            B, K, L = ctx_ids.shape
            # Flatten to encode
            flat_ctx_ids = ctx_ids.view(B * K, L)
            c_vec_flat = self.encode(flat_ctx_ids)  # (B*K, H)
            c_vec = c_vec_flat.view(B, K, -1)  # (B, K, H)

            # Dot Product with broadcasting
            # q_vec: (B, H) -> (B, 1, H)
            scores = torch.sum(q_vec.unsqueeze(1) * c_vec, dim=-1)  # (B, K)

        else:
            raise ValueError(f"ctx_ids must be 2D or 3D, got shape {ctx_ids.shape}")

        return scores


class ShallowCNNReader(nn.Module):
    """
    Shallow 1D-CNN for extracting Short Answer spans.
    """

    def __init__(self, vocab_size, padding_idx=0):
        super(ShallowCNNReader, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )

        # Stacked 1D Convolutions
        # Layer 1: Kernel Size 3
        # Padding = (Kernel - 1) / 2 to maintain length
        k1 = Config.READER_KERNEL_SIZES[0]
        p1 = (k1 - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels=Config.EMBEDDING_DIM,
            out_channels=Config.READER_FILTERS,
            kernel_size=k1,
            padding=p1,
        )

        # Layer 2: Kernel Size 5
        k2 = Config.READER_KERNEL_SIZES[1]
        p2 = (k2 - 1) // 2
        self.conv2 = nn.Conv1d(
            in_channels=Config.READER_FILTERS,
            out_channels=Config.READER_FILTERS,
            kernel_size=k2,
            padding=p2,
        )

        self.dropout = nn.Dropout(Config.READER_DROPOUT)
        self.activation = nn.ReLU()

        # Output Heads
        self.start_fc = nn.Linear(Config.READER_FILTERS, 1)
        self.end_fc = nn.Linear(Config.READER_FILTERS, 1)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (Batch, Seq_Len) - Concatenated Q and Context

        Returns:
            start_logits: (Batch, Seq_Len)
            end_logits: (Batch, Seq_Len)
        """
        # Embed
        x = self.embedding(input_ids)  # (B, L, E)

        # Permute for Conv1d: (B, Channels, Length)
        x = x.permute(0, 2, 1)  # (B, E, L)

        # Conv Layer 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Conv Layer 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Permute back: (B, L, Channels)
        x = x.permute(0, 2, 1)

        # Prediction Heads
        start_logits = self.start_fc(x).squeeze(-1)  # (B, L)
        end_logits = self.end_fc(x).squeeze(-1)  # (B, L)

        return start_logits, end_logits
