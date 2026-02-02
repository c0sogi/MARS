import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CompareAggregateRanker(nn.Module):
    """
    Ranker model that scores the relevance of a candidate paragraph to a question.
    Uses a Compare-Aggregate architecture:
    1. Encodes Question via Max-Pooling.
    2. Compares Question Vector with Paragraph Tokens (Element-wise).
    3. Aggregates interactions via Sum-Pooling.
    4. Scores via MLP.
    """

    def __init__(self, embedding_matrix, hidden_dim=128, dropout_prob=0.3):
        super(CompareAggregateRanker, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0,
        )

        # Comparison is element-wise, so input to aggregation is same size as embedding
        # We project this to a hidden dimension before scoring
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, question_ids, paragraph_ids):
        """
        Args:
            question_ids: (batch_size, q_len)
            paragraph_ids: (batch_size, p_len)
        Returns:
            scores: (batch_size, 1) - Relevance probability (logits before sigmoid if using BCEWithLogits,
                    but usually rankers output a score. We will output raw logits here for numerical stability
                    with BCEWithLogitsLoss, or apply sigmoid if needed).
                    Let's output logits.
        """
        # 1. Embedding
        # q_emb: (batch, q_len, embed_dim)
        q_emb = self.embedding(question_ids)
        # p_emb: (batch, p_len, embed_dim)
        p_emb = self.embedding(paragraph_ids)

        # 2. Question Encoding (Max-Pooling)
        # Mask padding (assuming 0 is pad)
        q_mask = (question_ids != 0).unsqueeze(-1).float()  # (batch, q_len, 1)
        # Replace padding with -inf for max pooling
        q_emb_masked = q_emb.masked_fill(q_mask == 0, -1e9)
        # q_vec: (batch, embed_dim)
        q_vec = torch.max(q_emb_masked, dim=1)[0]

        # 3. Comparison (Element-wise multiplication)
        # Broadcast q_vec to match p_len: (batch, 1, embed_dim) * (batch, p_len, embed_dim)
        # interaction: (batch, p_len, embed_dim)
        interaction = q_vec.unsqueeze(1) * p_emb

        # 4. Aggregation (Sum-Pooling)
        # Mask paragraph padding
        p_mask = (paragraph_ids != 0).unsqueeze(-1).float()
        interaction = interaction * p_mask
        # agg_vec: (batch, embed_dim)
        agg_vec = torch.sum(interaction, dim=1)

        # 5. Scoring
        logits = self.mlp(agg_vec)
        return logits


class DilatedConvReader(nn.Module):
    """
    Reader model that extracts short answer spans from a sequence.
    Uses Dilated Convolutions to capture long-range dependencies efficiently.
    """

    def __init__(
        self,
        embedding_matrix,
        hidden_dim=128,
        kernel_size=3,
        num_layers=4,
        dropout_prob=0.3,
    ):
        super(DilatedConvReader, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0,
        )

        # Project embedding to hidden dimension
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)

        self.conv_layers = nn.ModuleList()
        self.dilations = [2**i for i in range(num_layers)]  # 1, 2, 4, 8...

        for dilation in self.dilations:
            # Calculate padding to keep sequence length same (assuming odd kernel size)
            # padding = dilation * (kernel_size - 1) / 2
            padding = dilation * (kernel_size - 1) // 2

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                )
            )

        # Output heads for start and end logits
        self.start_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.end_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) - Concatenated Question + Paragraph
        Returns:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
        """
        # Embed and transpose for Conv1d (batch, embed_dim, seq_len)
        x = self.embedding(input_ids).transpose(1, 2)

        # Project to hidden dim
        x = self.input_proj(x)

        # Apply dilated convolutions with residual connections (optional but good practice,
        # though prompt description implies simple stack, standard ResNet blocks help gradient flow)
        # We will implement as a direct stack as per prompt description "passed through a stack".
        for layer in self.conv_layers:
            out = layer(x)
            x = x + out  # Adding residual connection for stability

        # Predict logits
        start_logits = self.start_head(x).squeeze(1)  # (batch, seq_len)
        end_logits = self.end_head(x).squeeze(1)  # (batch, seq_len)

        # Mask padding positions (force logits to -inf)
        mask = input_ids == 0
        start_logits = start_logits.masked_fill(mask, -1e9)
        end_logits = end_logits.masked_fill(mask, -1e9)

        return start_logits, end_logits
