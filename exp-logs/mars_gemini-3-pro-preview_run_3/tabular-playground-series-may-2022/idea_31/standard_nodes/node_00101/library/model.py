import torch
import torch.nn as nn
from library.config import NUM_STREAMS, EMBED_DIM, STREAM_CONFIGS, DROPOUT_RATES


class FEPFEStream(nn.Module):
    """
    A single independent stream for the FEPFE model.
    Contains its own independent embedding layers and MLP backbone.
    """

    def __init__(self, vocab_sizes, num_continuous, hidden_layers, dropout_rate):
        super(FEPFEStream, self).__init__()

        # Independent Embeddings for each of the 10 character positions
        # vocab_sizes is a list of length 10 containing the vocab size for each position
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_DIM)
                for vocab_size in vocab_sizes
            ]
        )

        # Calculate concatenated input dimension
        # Input = (10 categorical features * 16 dim) + continuous features
        input_dim = (len(vocab_sizes) * EMBED_DIM) + num_continuous

        # Construct MLP Backbone (Heterogeneous Funnel)
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Final projection to logit (binary classification)
        # Note: No sigmoid here, as we use BCEWithLogitsLoss later
        layers.append(nn.Linear(current_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: Tensor of shape (Batch, Num_Continuous)
            categorical: Tensor of shape (Batch, 10) containing integer indices
        """
        # 1. Process Embeddings
        embedded_parts = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select the i-th column of categorical indices
            col_indices = categorical[:, i]
            # Lookup embeddings -> (Batch, Embed_Dim)
            embedded_parts.append(emb_layer(col_indices))

        # 2. Flatten and Concatenate Embeddings
        # Shape: (Batch, 10 * Embed_Dim)
        x_emb = torch.cat(embedded_parts, dim=1)

        # 3. Early Fusion: Concatenate with Continuous Features
        # Shape: (Batch, Input_Dim)
        x = torch.cat([x_emb, continuous], dim=1)

        # 4. Pass through MLP
        logit = self.mlp(x)

        return logit


class FEPFEModel(nn.Module):
    """
    Frequency-Enhanced Parallel Funnel Ensemble (FEPFE).
    Consists of 5 independent streams instantiated within a single computational graph.
    """

    def __init__(self, vocab_sizes, num_continuous):
        super(FEPFEModel, self).__init__()

        self.streams = nn.ModuleList()

        # Instantiate 5 independent streams based on configuration
        for i in range(NUM_STREAMS):
            stream = FEPFEStream(
                vocab_sizes=vocab_sizes,
                num_continuous=num_continuous,
                hidden_layers=STREAM_CONFIGS[i],
                dropout_rate=DROPOUT_RATES[i],
            )
            self.streams.append(stream)

    def forward(self, continuous, categorical):
        """
        Forward pass through all 5 streams in parallel.

        Args:
            continuous: Tensor (Batch, N_cont)
            categorical: Tensor (Batch, 10)

        Returns:
            List of tensors, where each tensor is the logit output of a stream.
            Shape of each tensor: (Batch, 1)
        """
        stream_outputs = []
        for stream in self.streams:
            stream_outputs.append(stream(continuous, categorical))

        return stream_outputs
