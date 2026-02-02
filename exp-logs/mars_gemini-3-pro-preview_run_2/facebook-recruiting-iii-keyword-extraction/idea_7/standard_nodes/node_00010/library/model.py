import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WideDeepTextCNN(nn.Module):
    """
    Hybrid Wide-and-Deep Network with Multi-Scale CNN.

    Wide Component: Linear layer on sparse TF-IDF features.
    Deep Component: TextCNN on dense integer sequences.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_tags=Config.NUM_TAGS,
        tfidf_dim=Config.TFIDF_MAX_FEATURES,
        cnn_filters=Config.CNN_FILTERS,
        cnn_kernel_sizes=Config.CNN_KERNEL_SIZES,
        dropout=Config.DROPOUT,
    ):
        super(WideDeepTextCNN, self).__init__()

        # -------------------------------------------------------
        # Deep Component (TextCNN)
        # -------------------------------------------------------
        # Embedding Layer: Maps integer tokens to dense vectors
        # padding_idx=0 ensures the padding token vector remains zero
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )

        # Multi-Scale Convolutions
        # We use a ModuleList to hold parallel convolutional layers
        # Each convolution captures patterns of different lengths (n-grams)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=cnn_filters, kernel_size=k
                )
                for k in cnn_kernel_sizes
            ]
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Projection Layer for Deep Component
        # Input dimension is (Number of Filters * Number of Kernels)
        # This maps the extracted semantic features to the tag space
        deep_output_dim = cnn_filters * len(cnn_kernel_sizes)
        self.fc_deep = nn.Linear(deep_output_dim, num_tags)

        # -------------------------------------------------------
        # Wide Component
        # -------------------------------------------------------
        # Linear layer mapping TF-IDF features directly to tags
        # This captures explicit keyword matches
        self.fc_wide = nn.Linear(tfidf_dim, num_tags)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Dictionary containing:
                - 'wide': Tensor of shape (batch_size, tfidf_dim)
                - 'deep': Tensor of shape (batch_size, max_len)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_tags)
        """
        # -------------------------------------------------------
        # Wide Path
        # -------------------------------------------------------
        # Process sparse TF-IDF features
        x_wide = inputs["wide"]
        wide_logits = self.fc_wide(x_wide)

        # -------------------------------------------------------
        # Deep Path
        # -------------------------------------------------------
        # Process dense sequence features
        x_deep = inputs["deep"]

        # Embedding: (Batch, MaxLen) -> (Batch, MaxLen, EmbedDim)
        emb = self.embedding(x_deep)

        # Permute for Conv1d: (Batch, EmbedDim, MaxLen)
        # Conv1d expects channels (embedding dim) as the second dimension
        emb = emb.permute(0, 2, 1)

        # Apply Convolutions, ReLU, and Global Max Pooling
        conv_results = []
        for conv in self.convs:
            # Conv1d: (Batch, Filters, L_out)
            c = conv(emb)
            # ReLU Activation
            c = F.relu(c)
            # Global Max Pooling: (Batch, Filters, 1) -> (Batch, Filters)
            # Pool over the time dimension (dim 2) to find the strongest signal
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_results.append(p)

        # Concatenate pooled features from all kernel sizes
        # Shape: (Batch, Filters * NumKernels)
        cnn_out = torch.cat(conv_results, dim=1)

        # Apply Dropout
        cnn_out = self.dropout(cnn_out)

        # Map to Output Space
        deep_logits = self.fc_deep(cnn_out)

        # -------------------------------------------------------
        # Fusion
        # -------------------------------------------------------
        # Sum the logits from both components
        # This combines the "memorization" (Wide) and "generalization" (Deep) capabilities
        final_logits = wide_logits + deep_logits

        return final_logits
