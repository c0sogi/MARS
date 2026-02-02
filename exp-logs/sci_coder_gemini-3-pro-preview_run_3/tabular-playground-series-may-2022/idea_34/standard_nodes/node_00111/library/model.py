import torch
import torch.nn as nn
from library.config import Config


class ARPFEStream(nn.Module):
    """
    A single stream of the ARPFE architecture.
    Implements a Selective Wide-and-Deep topology with independent embeddings.
    """

    def __init__(self, vocab_sizes, cont_dim, layer_sizes, dropout):
        super().__init__()

        # 1. Independent Entity Embeddings
        # Each stream gets its own set of embeddings to learn diverse representations
        self.embeddings = nn.ModuleDict()
        self.cat_features = Config.get_all_cat_features()

        total_emb_dim = 0
        for feat in self.cat_features:
            # vocab_sizes is a dict {feature_name: size}
            num_embeddings = vocab_sizes[feat]
            emb_dim = Config.EMBEDDING_DIM

            self.embeddings[feat] = nn.Embedding(num_embeddings, emb_dim)
            total_emb_dim += emb_dim

        # 2. Deep Path (Funnel MLP)
        # Input: Continuous Features + Flattened Embeddings
        deep_input_dim = cont_dim + total_emb_dim

        layers = []
        current_dim = deep_input_dim

        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Final projection of Deep Path to a scalar logit
        layers.append(nn.Linear(current_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # 3. Wide Path (Selective Residual)
        # Input: Continuous Features ONLY (High-signal aggregates)
        # Bypasses embeddings to avoid noise in the linear path
        self.wide = nn.Linear(cont_dim, 1)

    def forward(self, cont_x, cat_x):
        """
        Args:
            cont_x: Tensor (Batch, Cont_Dim)
            cat_x: Tensor (Batch, Cat_Dim) - Indices
        """
        # 1. Process Embeddings
        emb_list = []
        # Iterate by index to match column order in cat_x
        for i, feat_name in enumerate(self.cat_features):
            indices = cat_x[:, i]
            emb = self.embeddings[feat_name](indices)
            emb_list.append(emb)

        # Flatten and Concatenate Embeddings
        cat_embeds = torch.cat(emb_list, dim=1)

        # 2. Deep Path Forward
        # Fused Input: Continuous + Embeddings
        deep_in = torch.cat([cont_x, cat_embeds], dim=1)
        deep_out = self.mlp(deep_in)

        # 3. Wide Path Forward
        # Input: Continuous ONLY
        wide_out = self.wide(cont_x)

        # 4. Aggregation (Sum)
        return deep_out + wide_out


class ARPFEModel(nn.Module):
    """
    Aggregate-Residual Parallel Funnel Ensemble (ARPFE).
    Contains 5 independent streams within a single computational graph.
    """

    def __init__(self, vocab_sizes):
        super().__init__()

        # Determine dimensions
        cont_features = Config.get_all_cont_features()
        cont_dim = len(cont_features)

        # Instantiate Independent Streams
        self.streams = nn.ModuleList()

        for stream_config in Config.STREAM_CONFIGS:
            stream = ARPFEStream(
                vocab_sizes=vocab_sizes,
                cont_dim=cont_dim,
                layer_sizes=stream_config["layers"],
                dropout=stream_config["dropout"],
            )
            self.streams.append(stream)

    def forward(self, cont_x, cat_x):
        """
        Computes logits for all 5 streams.

        Args:
            cont_x: Continuous features (Batch, Cont_Dim)
            cat_x: Categorical indices (Batch, Cat_Dim)

        Returns:
            Tensor of shape (Batch, 5) containing logits from each stream.
        """
        stream_outputs = []

        for stream in self.streams:
            out = stream(cont_x, cat_x)
            stream_outputs.append(out)

        # Concatenate along the feature dimension to get (Batch, 5)
        return torch.cat(stream_outputs, dim=1)
