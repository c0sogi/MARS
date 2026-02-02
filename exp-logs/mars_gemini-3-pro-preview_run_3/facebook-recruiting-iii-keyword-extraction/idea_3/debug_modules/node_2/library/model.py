import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TextCNN(nn.Module):
    """
    A 1D Convolutional Neural Network (TextCNN) for text classification.

    Architecture:
    1. Embedding Layer: Maps token IDs to dense vectors.
    2. Conv1d Layers: Parallel convolutions with different kernel sizes (e.g., 3, 4, 5)
       to capture local n-gram patterns.
    3. Global Max Pooling: Extracts the most salient feature from each convolution map.
    4. Concatenation: Combines features from all kernel sizes.
    5. Dropout & Linear: Regularization and mapping to output class logits.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=Config.TOP_K_TAGS,
        kernel_sizes=Config.KERNEL_SIZES,
        num_filters=Config.NUM_FILTERS,
        dropout=Config.DROPOUT,
        padding_idx=0,
    ):
        """
        Initializes the TextCNN model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the word embeddings.
            num_classes (int): Number of output classes (tags).
            kernel_sizes (list): List of kernel sizes for convolutions (e.g., [3, 4, 5]).
            num_filters (int): Number of filters per kernel size.
            dropout (float): Dropout probability.
            padding_idx (int): Index used for padding in the embedding layer.
        """
        super(TextCNN, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=padding_idx
        )

        # 2. Convolutional Layers
        # We use ModuleList to hold multiple Conv1d layers with different kernel sizes.
        # Input channels = embed_dim, Output channels = num_filters
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                )
                for k in kernel_sizes
            ]
        )

        # 3. Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # 4. Fully Connected Layer
        # The input dimension corresponds to the number of filters times the number of kernel sizes,
        # because we concatenate the max-pooled outputs of each kernel size.
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_len) containing token IDs.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # x shape: (batch_size, max_len)

        # Apply Embedding
        # Output shape: (batch_size, max_len, embed_dim)
        x = self.embedding(x)

        # Permute dimensions for Conv1d
        # Conv1d expects input shape: (batch_size, in_channels, sequence_length)
        # So we swap dimensions 1 and 2.
        # Output shape: (batch_size, embed_dim, max_len)
        x = x.permute(0, 2, 1)

        # Apply Convolutions + ReLU + Global Max Pooling
        conv_results = []
        for conv in self.convs:
            # Apply convolution
            # Shape: (batch_size, num_filters, L_out)
            feature_map = conv(x)

            # Apply ReLU activation
            feature_map = F.relu(feature_map)

            # Apply Global Max Pooling
            # We pool over the time dimension (dim=2) to get the single strongest feature per filter.
            # Kernel size for pooling is the length of the feature map after convolution.
            # Shape: (batch_size, num_filters)
            pooled = F.max_pool1d(
                feature_map, kernel_size=feature_map.shape[2]
            ).squeeze(2)

            conv_results.append(pooled)

        # Concatenate pooled features from all kernel sizes
        # Shape: (batch_size, num_filters * len(kernel_sizes))
        x = torch.cat(conv_results, dim=1)

        # Apply Dropout
        x = self.dropout(x)

        # Apply Linear Layer to get logits
        # Shape: (batch_size, num_classes)
        logits = self.fc(x)

        return logits
