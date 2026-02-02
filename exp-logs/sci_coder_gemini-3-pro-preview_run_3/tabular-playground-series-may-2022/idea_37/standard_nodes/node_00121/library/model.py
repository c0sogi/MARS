import torch
import torch.nn as nn
from library.config import Config


class Stream(nn.Module):
    """
    A single stream in the PIFE architecture implementing a pure Deep MLP topology.

    Attributes:
        embeddings (nn.ModuleList): Independent embedding layers for categorical features.
        deep_path (nn.Sequential): The 'Deep' MLP funnel processing fused inputs.
    """

    def __init__(self, n_cont, vocab_sizes, embed_dim, hidden_layers, dropout):
        super().__init__()

        # Independent Embeddings for this stream
        # We explicitly reject shared embeddings to ensure decorrelated representation learning
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Calculate input dimension for the Deep Path
        # Input Fusion: Continuous + Flattened Embeddings
        n_cat_flat = len(vocab_sizes) * embed_dim
        deep_input_dim = n_cont + n_cat_flat

        # Deep Path (The Funnel)
        # Topology: Input -> Hidden 1 -> ... -> Output
        layers = []
        in_dim = deep_input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())  # Standard ReLU exclusively
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection of Deep Path
        layers.append(nn.Linear(in_dim, 1))
        self.deep_path = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat):
        # 1. Process Embeddings
        # x_cat shape: (Batch, n_cat)
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)  # Flatten embeddings

        # 2. Deep Path Input Fusion
        x_deep_in = torch.cat([x_cont, x_emb], dim=1)

        # 3. Compute Paths
        out_deep = self.deep_path(x_deep_in)

        # 4. Return Deep Path Output
        return out_deep


class DAR_PE_Model(nn.Module):
    """
    Deep Aggregate-Residual Parallel Ensemble (DAR-PE).

    Consists of 5 independent streams with heterogeneous capacity and regularization.
    """

    def __init__(self, n_cont, vocab_sizes):
        super().__init__()
        self.streams = nn.ModuleList()

        # Instantiate streams based on the heterogeneous configurations defined in Config
        # Streams 1 & 2 (Anchors), 3 & 4 (Capacity Variants), 5 (Conservative)
        for cfg in Config.STREAM_CONFIGS:
            self.streams.append(
                Stream(
                    n_cont=n_cont,
                    vocab_sizes=vocab_sizes,
                    embed_dim=Config.EMBED_DIM,
                    hidden_layers=cfg["layers"],
                    dropout=cfg["dropout"],
                )
            )

    def forward(self, x_cont, x_cat):
        """
        Forward pass for all independent streams.

        Returns:
            list[torch.Tensor]: A list of 5 output tensors (logits), one per stream.
        """
        outputs = []
        for stream in self.streams:
            outputs.append(stream(x_cont, x_cat))

        return outputs
