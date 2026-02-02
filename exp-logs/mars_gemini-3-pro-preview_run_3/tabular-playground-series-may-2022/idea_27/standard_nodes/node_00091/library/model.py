import torch
import torch.nn as nn


class FunnelStream(nn.Module):
    """
    Defines a single stream of the PIFE model.
    Contains independent embeddings and a Deep Funnel (MLP).
    Cite {lesson_node_00089}: Removed explicit interaction layers to avoid redundancy and improve speed.
    """

    def __init__(
        self, num_cont, cat_cardinalities, hidden_dims, dropout_rate, emb_dim=16
    ):
        super().__init__()

        # 1. Independent Embeddings
        # Each stream has its own set of embeddings to ensure decorrelated representation learning.
        # Cite {lesson_node_00065}: Independent embeddings maximize ensemble diversity.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c, embedding_dim=emb_dim)
                for c in cat_cardinalities
            ]
        )

        # 2. Deep Path (The Funnel)
        # Input: Continuous features + Flattened embeddings
        deep_input_dim = num_cont + (len(cat_cardinalities) * emb_dim)

        layers = []
        in_dim = deep_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.deep_mlp = nn.Sequential(*layers)
        self.deep_head = nn.Linear(in_dim, 1)

    def forward(self, x_cont, x_cat):
        # Retrieve embeddings
        # x_cat shape: [batch_size, num_cat_features]
        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            embs.append(emb_layer(x_cat[:, i]))

        # --- Deep Path ---
        # Concatenate all embeddings: [batch, num_cat * emb_dim]
        all_embs_flat = torch.cat(embs, dim=1)
        # Concatenate with continuous: [batch, num_cont + num_cat * emb_dim]
        deep_in = torch.cat([x_cont, all_embs_flat], dim=1)
        deep_out = self.deep_head(self.deep_mlp(deep_in))

        return deep_out


class PIFEModel(nn.Module):
    """
    Parallel Independent Funnel Ensemble (PIFE).
    Consists of multiple independent FunnelStreams instantiated within a single computational graph.
    Cite {lesson_node_00064}: Parallel in-network ensembling.
    """

    def __init__(self, num_cont, cat_cardinalities, stream_configs):
        super().__init__()
        self.streams = nn.ModuleList()

        for config in stream_configs:
            stream = FunnelStream(
                num_cont=num_cont,
                cat_cardinalities=cat_cardinalities,
                hidden_dims=config["hidden_dims"],
                dropout_rate=config["dropout"],
            )
            self.streams.append(stream)

    def forward(self, x_cont, x_cat):
        # Execute each stream independently
        outputs = [stream(x_cont, x_cat) for stream in self.streams]

        # Concatenate outputs: [batch_size, num_streams]
        return torch.cat(outputs, dim=1)
