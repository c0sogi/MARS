import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config

# Set fixed random seed for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class SiameseTextCNN(nn.Module):
    """
    Long Answer Ranker using a Siamese Multi-Scale 1D-Convolutional Network.

    Architecture:
    1. Shared Embedding Layer
    2. Multi-Scale 1D Convolutions (Bigrams, Trigrams, etc.)
    3. Max-Over-Time Pooling
    4. Heuristic Combination (Concat, Diff, Product)
    5. Binary Classification MLP
    """

    def __init__(self):
        super(SiameseTextCNN, self).__init__()

        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.filters = Config.CNN_FILTERS
        self.kernel_sizes = Config.CNN_KERNEL_SIZES
        self.hidden_dim = Config.RANKER_HIDDEN_DIM
        self.dropout_rate = Config.RANKER_DROPOUT

        # Shared Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Encoder: Parallel 1D Convolutions
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.embed_dim, out_channels=self.filters, kernel_size=k
                )
                for k in self.kernel_sizes
            ]
        )

        # Dimension after pooling and concatenating filters
        self.encoder_dim = self.filters * len(self.kernel_sizes)

        # Classifier Input Dimension: [u; v; |u-v|; u*v] -> 4 vectors
        self.classifier_input_dim = 4 * self.encoder_dim

        # Classification MLP
        self.fc1 = nn.Linear(self.classifier_input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward_one(self, x):
        """
        Encodes a single sequence (Question or Paragraph) into a fixed vector.
        """
        # Embed: (Batch, Seq_Len, Embed_Dim)
        emb = self.embedding(x)

        # Transpose for Conv1d: (Batch, Embed_Dim, Seq_Len)
        emb = emb.permute(0, 2, 1)

        # Apply Convolutions + ReLU
        # Output of each conv: (Batch, Filters, L_out)
        conved = [F.relu(conv(emb)) for conv in self.convs]

        # Max-Over-Time Pooling
        # Output: (Batch, Filters)
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]

        # Concatenate: (Batch, Filters * Num_Kernels)
        encoded = torch.cat(pooled, dim=1)
        return encoded

    def forward(self, question, paragraph):
        """
        Forward pass for the Siamese Network.
        Returns logits for binary classification (Match vs No-Match).
        """
        # Encode inputs independently
        u = self.forward_one(question)
        v = self.forward_one(paragraph)

        # Heuristic Matching Function
        diff = torch.abs(u - v)
        prod = u * v

        # Concatenate representations
        combined = torch.cat([u, v, diff, prod], dim=1)

        # MLP Classifier
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        # Return shape (Batch,)
        return logits.squeeze(1)


class AttentionMLPReader(nn.Module):
    """
    Short Answer Reader using Context-Query Attention and Time-Distributed MLP.

    Architecture:
    1. Shared Embedding Layer
    2. Context-Query Attention (Weighted sum of Question embeddings for each Paragraph token)
    3. Fusion (Concat Paragraph embedding + Context vector)
    4. Time-Distributed MLP for Start/End logits
    """

    def __init__(self):
        super(AttentionMLPReader, self).__init__()

        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.hidden_dim = Config.READER_HIDDEN_DIM
        self.dropout_rate = Config.READER_DROPOUT

        # Shared Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Fusion Dimension: Paragraph Embedding + Context Vector
        self.fusion_dim = self.embed_dim * 2

        # Extraction MLP (Shared across time steps)
        self.fc1 = nn.Linear(self.fusion_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Output Heads
        self.start_classifier = nn.Linear(self.hidden_dim, 1)
        self.end_classifier = nn.Linear(self.hidden_dim, 1)

    def forward(self, question, paragraph):
        """
        Forward pass for the Reader.
        Returns start_logits and end_logits for each token in the paragraph.
        """
        # Embeddings
        # Q: (Batch, Q_Len, Embed_Dim)
        # P: (Batch, P_Len, Embed_Dim)
        q_emb = self.embedding(question)
        p_emb = self.embedding(paragraph)

        # --- Context-Query Attention ---

        # Compute Similarity Matrix: P x Q^T
        # (Batch, P_Len, Embed_Dim) x (Batch, Embed_Dim, Q_Len) -> (Batch, P_Len, Q_Len)
        scores = torch.bmm(p_emb, q_emb.transpose(1, 2))

        # Compute Attention Weights (Softmax over Question dimension)
        # (Batch, P_Len, Q_Len)
        attn_weights = F.softmax(scores, dim=-1)

        # Compute Context Vectors (Weighted sum of Question embeddings)
        # (Batch, P_Len, Q_Len) x (Batch, Q_Len, Embed_Dim) -> (Batch, P_Len, Embed_Dim)
        context = torch.bmm(attn_weights, q_emb)

        # --- Fusion ---

        # Concatenate Paragraph embeddings with Context vectors
        # (Batch, P_Len, 2 * Embed_Dim)
        combined = torch.cat([p_emb, context], dim=-1)

        # --- Prediction ---

        # Apply MLP to every token position
        h = self.fc1(combined)
        h = F.relu(h)
        h = self.dropout(h)

        # Predict Logits
        # (Batch, P_Len)
        start_logits = self.start_classifier(h).squeeze(-1)
        end_logits = self.end_classifier(h).squeeze(-1)

        return start_logits, end_logits
