import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Applies attention mechanism to aggregate features over the sequence dimension.
    Input: (Batch, Channels, Seq_Len)
    Output: (Batch, Channels)
    """

    def __init__(self, input_dim, attention_dim):
        super(AttentionPooling, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        # Permute to (batch_size, seq_len, channels) for Linear layer
        x_perm = x.permute(0, 2, 1)

        # Calculate attention scores
        # (batch_size, seq_len, 1)
        scores = self.attention_layer(x_perm)

        # Calculate softmax weights over the sequence dimension (dim=1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum
        # (batch_size, seq_len, channels) * (batch_size, seq_len, 1) -> sum over seq_len
        # Result shape: (batch_size, channels)
        context = torch.sum(x_perm * weights, dim=1)

        return context


class WideAndDeepModel(nn.Module):
    """
    Hybrid Wide-and-Deep architecture for Tag Prediction.
    - Wide: Linear model on TF-IDF features.
    - Deep: Multi-scale TextCNN with Attention Pooling on Token IDs.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        wide_dim=Config.TFIDF_MAX_FEATURES,
        num_classes=Config.NUM_CLASSES,
        filter_sizes=Config.FILTER_SIZES,
        num_filters=Config.NUM_FILTERS,
        attention_dim=Config.ATTENTION_DIM,
        dropout=Config.DROPOUT,
    ):
        super(WideAndDeepModel, self).__init__()

        # -------------------------------------------------------
        # Deep Component (TextCNN + Attention)
        # -------------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        # Parallel Convolutional Layers
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    padding="valid",  # Standard TextCNN approach
                )
                for fs in filter_sizes
            ]
        )

        # Attention Pooling for each Conv layer
        self.attentions = nn.ModuleList(
            [AttentionPooling(num_filters, attention_dim) for _ in filter_sizes]
        )

        self.dropout = nn.Dropout(dropout)

        # Projection from concatenated deep features to output classes
        deep_output_dim = num_filters * len(filter_sizes)
        self.deep_fc = nn.Linear(deep_output_dim, num_classes)

        # -------------------------------------------------------
        # Wide Component (Linear)
        # -------------------------------------------------------
        self.wide_fc = nn.Linear(wide_dim, num_classes)

    def forward(self, deep_input, wide_input):
        """
        Args:
            deep_input: Tensor of shape (Batch, Max_Len) containing token IDs.
            wide_input: Tensor of shape (Batch, Wide_Dim) containing TF-IDF features.
        Returns:
            logits: Tensor of shape (Batch, Num_Classes)
        """
        # -------------------------------------------------------
        # Deep Path
        # -------------------------------------------------------
        # Embedding: (Batch, Max_Len, Emb_Dim)
        emb = self.embedding(deep_input)

        # Permute for Conv1d: (Batch, Emb_Dim, Max_Len)
        emb = emb.permute(0, 2, 1)

        deep_features = []
        for conv, attn in zip(self.convs, self.attentions):
            # Convolution: (Batch, Num_Filters, L_out)
            c = conv(emb)
            c = F.relu(c)

            # Attention Pooling: (Batch, Num_Filters)
            p = attn(c)
            deep_features.append(p)

        # Concatenate pooled features: (Batch, Num_Filters * len(filter_sizes))
        deep_concat = torch.cat(deep_features, dim=1)

        # Dropout and Projection
        deep_concat = self.dropout(deep_concat)
        deep_logits = self.deep_fc(deep_concat)

        # -------------------------------------------------------
        # Wide Path
        # -------------------------------------------------------
        # Linear projection of sparse features
        wide_logits = self.wide_fc(wide_input)

        # -------------------------------------------------------
        # Fusion
        # -------------------------------------------------------
        # Sum logits (equivalent to summing probabilities in log-space if using Softmax,
        # but here we use Sigmoid independently for multi-label)
        total_logits = deep_logits + wide_logits

        return total_logits
