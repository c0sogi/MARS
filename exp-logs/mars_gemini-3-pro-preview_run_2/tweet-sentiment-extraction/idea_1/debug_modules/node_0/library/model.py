import torch
import torch.nn as nn
from library.config import (
    EMBED_DIM,
    SENTIMENT_EMBED_DIM,
    HIDDEN_DIM,
    DROPOUT,
    SENTIMENT_MAP,
)


class BiGRUPointerNetwork(nn.Module):
    """
    A Bidirectional GRU Pointer Network for sentiment span extraction.

    Architecture:
    1. Word Embeddings & Sentiment Embeddings
    2. Concatenation: Word Emb + Sentiment Emb (repeated for seq len)
    3. Bi-Directional GRU
    4. Two Linear Heads: Start Index Logits, End Index Logits
    """

    def __init__(self, vocab_size):
        super(BiGRUPointerNetwork, self).__init__()

        # 1. Embeddings
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=EMBED_DIM, padding_idx=0
        )

        self.sentiment_embedding = nn.Embedding(
            num_embeddings=len(SENTIMENT_MAP), embedding_dim=SENTIMENT_EMBED_DIM
        )

        # 2. Recurrent Layer
        # Input size is sum of word embedding and sentiment embedding dimensions
        input_size = EMBED_DIM + SENTIMENT_EMBED_DIM

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Regularization
        self.dropout = nn.Dropout(DROPOUT)

        # 4. Output Heads
        # The GRU is bidirectional, so the hidden size is doubled
        gru_output_dim = HIDDEN_DIM * 2

        self.start_head = nn.Linear(gru_output_dim, 1)
        self.end_head = nn.Linear(gru_output_dim, 1)

    def forward(self, input_ids, sentiment_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            sentiment_ids: (batch_size)
            attention_mask: (batch_size, seq_len) - Not strictly used by GRU but kept for interface consistency

        Returns:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
        """
        # Get Word Embeddings -> (batch_size, seq_len, embed_dim)
        word_emb = self.word_embedding(input_ids)

        # Get Sentiment Embeddings -> (batch_size, sentiment_embed_dim)
        sent_emb = self.sentiment_embedding(sentiment_ids)

        # Expand sentiment embeddings to match sequence length
        # (batch_size, 1, sentiment_embed_dim)
        sent_emb = sent_emb.unsqueeze(1)
        # (batch_size, seq_len, sentiment_embed_dim)
        sent_emb = sent_emb.expand(-1, input_ids.size(1), -1)

        # Concatenate embeddings -> (batch_size, seq_len, embed_dim + sentiment_embed_dim)
        embeddings = torch.cat([word_emb, sent_emb], dim=2)

        # Apply Dropout to inputs
        embeddings = self.dropout(embeddings)

        # Pass through Bi-GRU
        # output: (batch_size, seq_len, hidden_dim * 2)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        gru_out, _ = self.gru(embeddings)

        # Apply Dropout to hidden states
        gru_out = self.dropout(gru_out)

        # Predict Start and End logits
        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        start_logits = self.start_head(gru_out).squeeze(-1)
        end_logits = self.end_head(gru_out).squeeze(-1)

        return start_logits, end_logits
