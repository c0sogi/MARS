import torch
import torch.nn as nn
from library.config import Config


class WideDeepStream(nn.Module):
    """
    A single stream implementing the Wide-and-Deep topology with a Funnel MLP.

    Structure:
    - Inputs: Continuous features + Categorical features
    - Embeddings: Independent Entity Embeddings for each categorical feature.
    - Fusion: Concatenation of flattened embeddings and continuous features.
    - Wide Path: Linear skip connection (Input -> Output).
    - Deep Path: Funnel MLP (Input -> Hidden Layers -> Output).
    - Aggregation: Sum(Wide, Deep).
    """

    def __init__(
        self, vocab_sizes, num_continuous, embedding_dim, hidden_layers, dropout_rate
    ):
        super(WideDeepStream, self).__init__()

        # 1. Independent Entity Embeddings
        # vocab_sizes is a dict {col_name: size}. Order is preserved from FeatureEngineer.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
                for size in vocab_sizes.values()
            ]
        )

        # Calculate fused input dimension
        self.num_categorical = len(vocab_sizes)
        self.fused_dim = num_continuous + (self.num_categorical * embedding_dim)

        # 2. Wide Path (Linear Skip)
        # Direct projection from input to output logit
        self.wide_linear = nn.Linear(self.fused_dim, 1)

        # 3. Deep Path (Funnel MLP)
        layers = []
        in_dim = self.fused_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Final projection of Deep Path to scalar logit
        layers.append(nn.Linear(in_dim, 1))

        self.deep_mlp = nn.Sequential(*layers)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: Tensor (batch, num_continuous)
            categorical: Tensor (batch, num_categorical) - LongTensor
        """
        # Process Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select the i-th categorical column
            emb_list.append(emb_layer(categorical[:, i]))

        # Concatenate all embeddings: (batch, num_cat * emb_dim)
        x_emb = torch.cat(emb_list, dim=1)

        # Early Fusion: Concatenate embeddings with continuous features
        x_fused = torch.cat([x_emb, continuous], dim=1)

        # Wide Path Forward
        out_wide = self.wide_linear(x_fused)

        # Deep Path Forward
        out_deep = self.deep_mlp(x_fused)

        # Aggregation: Sum of logits
        return out_wide + out_deep


class RPFEModel(nn.Module):
    """
    Residual-Parallel Funnel Ensemble (RPFE)

    Contains 5 independent WideDeepStream instances with heterogeneous configurations
    as defined in Config.
    """

    def __init__(self, vocab_sizes, num_continuous):
        super(RPFEModel, self).__init__()

        self.streams = nn.ModuleList()

        # Initialize streams based on Config
        for stream_conf in Config.STREAM_CONFIGS:
            stream = WideDeepStream(
                vocab_sizes=vocab_sizes,
                num_continuous=num_continuous,
                embedding_dim=Config.EMBEDDING_DIM,
                hidden_layers=stream_conf["layers"],
                dropout_rate=stream_conf["dropout"],
            )
            self.streams.append(stream)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: Tensor (batch, num_continuous)
            categorical: Tensor (batch, num_categorical)

        Returns:
            Tensor (batch, num_streams): The logits from each independent stream.
        """
        stream_outputs = []

        for stream in self.streams:
            out = stream(continuous, categorical)
            stream_outputs.append(out)

        # Concatenate outputs along the feature dimension (columns)
        # Shape becomes (batch_size, num_streams)
        return torch.cat(stream_outputs, dim=1)
