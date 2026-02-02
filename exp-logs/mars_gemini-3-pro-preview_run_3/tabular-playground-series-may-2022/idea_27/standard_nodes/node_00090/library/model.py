import torch
import torch.nn as nn


class FunnelStream(nn.Module):
    """
    Defines a single stream of the PIFE model.
    Standard Deep MLP with independent embeddings.
    """

    def __init__(
        self, num_cont, cat_cardinalities, hidden_dims, dropout_rate, emb_dim=16
    ):
        super().__init__()

        # Independent Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c, embedding_dim=emb_dim)
                for c in cat_cardinalities
            ]
        )

        # Deep Path
        input_dim = num_cont + (len(cat_cardinalities) * emb_dim)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x_cont, x_cat):
        # Retrieve embeddings
        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            embs.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings
        all_embs_flat = torch.cat(embs, dim=1)

        # Concatenate with continuous
        x = torch.cat([x_cont, all_embs_flat], dim=1)

        return self.head(self.mlp(x))


class PIFEModel(nn.Module):
    """
    Parallel Independent Funnel Ensemble (PIFE).
    Consists of multiple independent FunnelStreams instantiated within a single computational graph.
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
