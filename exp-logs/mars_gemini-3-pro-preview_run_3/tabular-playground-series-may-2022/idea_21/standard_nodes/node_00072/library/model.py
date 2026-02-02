import torch
import torch.nn as nn
from library.config import Config


class StreamBlock(nn.Module):
    """
    A single stream in the RSPFE architecture.
    Contains independent embeddings and a specific MLP configuration.
    """

    def __init__(
        self, vocab_sizes, embedding_dim, num_cont_features, layer_sizes, dropout_rate
    ):
        super(StreamBlock, self).__init__()

        # Independent Embeddings for this stream
        # Each categorical feature gets its own embedding layer
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate input dimension for the MLP
        # (Number of categorical features * Embedding Dim) + Number of continuous features
        input_dim = (len(vocab_sizes) * embedding_dim) + num_cont_features

        # Build the Funnel MLP Backbone
        layers = []
        in_dim = input_dim

        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())  # Standard ReLU exclusively
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Final projection to a single neuron
        # We avoid tapering to 64 units; we project directly from the last hidden layer (>=128)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: LongTensor of shape (batch_size, num_cat_features)
            x_cont: FloatTensor of shape (batch_size, num_cont_features)
        """
        # 1. Embed Categorical Features
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select the i-th categorical column
            col_idx = x_cat[:, i]
            embedded_features.append(emb_layer(col_idx))

        # 2. Concatenate Embeddings
        # Shape: (batch_size, num_cat_features * embedding_dim)
        x_emb = torch.cat(embedded_features, dim=1)

        # 3. Early Fusion
        # Concatenate embeddings with continuous features
        x = torch.cat([x_emb, x_cont], dim=1)

        # 4. MLP Backbone
        x = self.mlp(x)

        # 5. Output Projection (Logits)
        return self.head(x)


class RSPFEModel(nn.Module):
    """
    Regularization-Spectrum Parallel Funnel Ensemble (RSPFE).
    Consists of 5 independent streams with varying capacity and regularization.
    """

    def __init__(self, vocab_sizes):
        super(RSPFEModel, self).__init__()

        self.streams = nn.ModuleList()

        num_cont_features = len(Config.CONT_FEATURES)
        embedding_dim = Config.EMBEDDING_DIM

        # Instantiate each stream based on the configuration
        for stream_config in Config.STREAMS:
            stream = StreamBlock(
                vocab_sizes=vocab_sizes,
                embedding_dim=embedding_dim,
                num_cont_features=num_cont_features,
                layer_sizes=stream_config["layers"],
                dropout_rate=stream_config["dropout"],
            )
            self.streams.append(stream)

    def forward(self, x_cat, x_cont):
        """
        Forward pass for the ensemble.
        Returns logits for all 5 streams.

        Returns:
            Tensor of shape (batch_size, 5)
        """
        outputs = []
        for stream in self.streams:
            outputs.append(stream(x_cat, x_cont))

        # Concatenate outputs along the feature dimension
        # Result shape: (batch_size, 5)
        return torch.cat(outputs, dim=1)
