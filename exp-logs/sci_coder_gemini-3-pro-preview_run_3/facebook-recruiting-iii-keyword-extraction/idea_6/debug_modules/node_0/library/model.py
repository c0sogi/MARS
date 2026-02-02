import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class WideDeepTextCNN(nn.Module):
    """
    Wide-and-Deep TextCNN Architecture.

    Combines a linear 'Wide' component (Bag-of-Words) with a 'Deep' TextCNN component.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the dense embeddings for the deep component.
        num_classes (int): Number of output classes (tags).
        kernel_sizes (list): List of kernel sizes for the convolution layers.
        num_filters (int): Number of filters per kernel size.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_classes=config.TOP_K_TAGS,
        kernel_sizes=config.KERNEL_SIZES,
        num_filters=config.NUM_FILTERS,
        dropout=config.DROPOUT,
    ):
        super(WideDeepTextCNN, self).__init__()

        # =====================================================================
        # Wide Component (Linear / Memorization)
        # =====================================================================
        # Maps token IDs directly to class logits, summing contributions.
        # This acts effectively as a Logistic Regression on Bag-of-Words features.
        # padding_idx=0 ensures the padding token contributes 0 to the sum.
        self.wide = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=num_classes,
            mode="sum",
            padding_idx=0,
        )

        # =====================================================================
        # Deep Component (TextCNN / Generalization)
        # =====================================================================
        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )

        # 2. Convolutional Layers
        # Parallel 1D convolutions to capture local n-gram patterns.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,  # Padding to handle edge cases
                )
                for k in kernel_sizes
            ]
        )

        # 3. Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # 4. Fully Connected Layer
        # Projects concatenated pooled features to output space.
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
                              containing integer token IDs.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # =====================================================================
        # Wide Stream Forward
        # =====================================================================
        # x shape: (batch_size, seq_len) -> wide_out shape: (batch_size, num_classes)
        wide_out = self.wide(x)

        # =====================================================================
        # Deep Stream Forward
        # =====================================================================
        # 1. Embedding
        # Output: (batch_size, seq_len, embed_dim)
        emb = self.embedding(x)

        # 2. Transpose for Conv1d
        # Conv1d expects (batch_size, channels, length)
        # Output: (batch_size, embed_dim, seq_len)
        emb = emb.transpose(1, 2)

        # 3. Convolutions + Activation + Pooling
        conv_outs = []
        for conv in self.convs:
            # Conv: (batch_size, num_filters, L_out)
            feat = conv(emb)
            # Activation
            feat = F.relu(feat)
            # Global Max Pooling over time dimension
            # Output: (batch_size, num_filters)
            feat = F.max_pool1d(feat, feat.size(2)).squeeze(2)
            conv_outs.append(feat)

        # 4. Concatenate Features
        # Output: (batch_size, num_filters * len(kernel_sizes))
        deep_feat = torch.cat(conv_outs, dim=1)

        # 5. Dropout
        deep_feat = self.dropout(deep_feat)

        # 6. Linear Projection
        # Output: (batch_size, num_classes)
        deep_out = self.fc(deep_feat)

        # =====================================================================
        # Fusion
        # =====================================================================
        # Sum logits from both streams (Wide + Deep)
        logits = wide_out + deep_out

        return logits
