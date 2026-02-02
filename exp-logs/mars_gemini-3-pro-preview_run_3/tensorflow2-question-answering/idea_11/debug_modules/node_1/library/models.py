import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.modules import HistogramBinning, QRNNLayer


class HistogramRanker(nn.Module):
    """
    Histogram-Based Matching Ranker.
    Uses interaction histograms between query and document embeddings to predict relevance.
    """

    def __init__(self, embedding_matrix):
        super(HistogramRanker, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape
        # Initialize embedding layer with pre-trained weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )

        # Histogram Binning Layer
        self.hist_binning = HistogramBinning(num_bins=Config.HISTOGRAM_BINS)

        # MLP to score the histogram vectors (Matching aggregation)
        # Maps the bin counts to a relevance score for each query term
        self.mlp = nn.Sequential(
            nn.Linear(Config.HISTOGRAM_BINS, Config.RANKER_HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(Config.RANKER_HIDDEN_DIM, 1),
        )

        # Gating network to weight the importance of each query term
        self.gating = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softmax(dim=1))

        self.dropout = nn.Dropout(Config.RANKER_DROPOUT)

    def forward(self, q_ids, doc_ids):
        """
        Args:
            q_ids: Tensor (batch_size, q_len)
            doc_ids: Tensor (batch_size, doc_len)

        Returns:
            score: Tensor (batch_size,) containing relevance scores
        """
        # 1. Embedding
        q_embed = self.embedding(q_ids)  # (batch, q_len, dim)
        d_embed = self.embedding(doc_ids)  # (batch, d_len, dim)

        q_embed = self.dropout(q_embed)
        d_embed = self.dropout(d_embed)

        # 2. Histogram Binning
        # Output: (batch, q_len, num_bins)
        hist = self.hist_binning(q_embed, d_embed)

        # Log-transformation to dampen the effect of large counts
        hist = torch.log(hist + 1e-6)

        # 3. Scoring per query term
        # (batch, q_len, num_bins) -> (batch, q_len, 1)
        term_scores = self.mlp(hist)

        # 4. Gating (Term Importance)
        # (batch, q_len, dim) -> (batch, q_len, 1)
        term_weights = self.gating(q_embed)

        # 5. Aggregation
        # Weighted sum of term scores
        # (batch, q_len, 1) -> sum -> (batch, 1)
        final_score = torch.sum(term_scores * term_weights, dim=1).squeeze(-1)

        return final_score


class QRNNReader(nn.Module):
    """
    Quasi-Recurrent Neural Network (QRNN) Reader.
    Extracts short answer spans from concatenated query and document sequences.
    """

    def __init__(self, embedding_matrix):
        super(QRNNReader, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )

        # Stack of QRNN Layers
        self.layers = nn.ModuleList()
        input_size = embed_dim

        for _ in range(Config.QRNN_NUM_LAYERS):
            self.layers.append(
                QRNNLayer(
                    input_size=input_size,
                    hidden_size=Config.QRNN_HIDDEN_DIM,
                    kernel_size=Config.QRNN_KERNEL_SIZE,
                    dropout=Config.QRNN_DROPOUT,
                )
            )
            input_size = Config.QRNN_HIDDEN_DIM

        # Output heads for Start and End positions
        self.start_head = nn.Linear(Config.QRNN_HIDDEN_DIM, 1)
        self.end_head = nn.Linear(Config.QRNN_HIDDEN_DIM, 1)

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor (batch_size, seq_len)

        Returns:
            start_logits: Tensor (batch_size, seq_len)
            end_logits: Tensor (batch_size, seq_len)
        """
        # Embed: (batch, seq_len, dim)
        x = self.embedding(input_ids)

        # Pass through QRNN stack
        for layer in self.layers:
            x = layer(x)

        # x is now (batch, seq_len, hidden_dim)

        # Predict logits
        start_logits = self.start_head(x).squeeze(-1)  # (batch, seq_len)
        end_logits = self.end_head(x).squeeze(-1)  # (batch, seq_len)

        return start_logits, end_logits
