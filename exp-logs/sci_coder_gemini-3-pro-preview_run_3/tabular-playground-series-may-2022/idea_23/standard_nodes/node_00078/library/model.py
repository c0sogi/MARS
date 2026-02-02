import torch
import torch.nn as nn
from library.config import Config


class FunnelStream(nn.Module):
    """
    A single stream of the Anchor-Variant Parallel Funnel Ensemble.
    Each stream maintains its own independent embedding space and MLP parameters.
    """

    def __init__(self, vocab_sizes, num_cont_features, hidden_dims, dropout_rate):
        super().__init__()

        # Independent Entity Embeddings
        # We create a separate embedding layer for each categorical feature
        # This ensures that each stream learns a unique representation of the categorical manifold
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(vs), embedding_dim=Config.EMBEDDING_DIM)
                for vs in vocab_sizes
            ]
        )

        # Calculate the total input dimension after flattening embeddings and concatenation
        # Input = Continuous Features + (Num Categorical Features * Embedding Dim)
        self.total_input_dim = num_cont_features + (
            len(vocab_sizes) * Config.EMBEDDING_DIM
        )

        # Construct the Funnel MLP Backbone
        layers = []
        in_dim = self.total_input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Final projection to a single logit
        # We project from the last hidden layer (e.g., 128) directly to 1
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: Tensor of shape (Batch, Num_Cont)
            x_cat: Tensor of shape (Batch, Num_Cat) containing integer indices
        """
        # 1. Embed Categorical Features
        # Iterate through columns and apply corresponding embedding layer
        # x_cat[:, i] gets the i-th categorical feature for the batch
        embedded_features = [
            embed(x_cat[:, i]) for i, embed in enumerate(self.embeddings)
        ]

        # 2. Flatten and Concatenate Embeddings
        # List of (B, Emb_Dim) -> (B, Num_Cat * Emb_Dim)
        x_emb_flat = torch.cat(embedded_features, dim=1)

        # 3. Early Fusion
        # Concatenate continuous features with flattened embeddings
        x = torch.cat([x_cont, x_emb_flat], dim=1)

        # 4. Pass through MLP
        return self.mlp(x)


class AVPFEModel(nn.Module):
    """
    Anchor-Variant Parallel Funnel Ensemble (AV-PFE).
    Consists of 5 independent streams defined in Config.STREAM_CONFIGS.
    """

    def __init__(self, vocab_sizes, num_cont_features):
        super().__init__()

        self.streams = nn.ModuleList()

        # Instantiate the 5 streams based on the configuration
        # 0-1: Anchors (Standard, Drop 0.20)
        # 2: Capacity Variant (Wide, Drop 0.25)
        # 3: Aggressive Variant (Standard, Drop 0.10)
        # 4: Conservative Variant (Standard, Drop 0.30)
        for i in range(Config.NUM_STREAMS):
            stream_config = Config.STREAM_CONFIGS[i]

            stream = FunnelStream(
                vocab_sizes=vocab_sizes,
                num_cont_features=num_cont_features,
                hidden_dims=stream_config["hidden_dims"],
                dropout_rate=stream_config["dropout"],
            )
            self.streams.append(stream)

    def forward(self, x_cont, x_cat):
        """
        Returns:
            Tensor of shape (Batch, Num_Streams) containing logits for each stream.
        """
        # Execute all streams in parallel
        stream_outputs = [stream(x_cont, x_cat) for stream in self.streams]

        # Concatenate outputs along the feature dimension
        # Each stream outputs (B, 1), result is (B, 5)
        return torch.cat(stream_outputs, dim=1)
