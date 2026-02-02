import torch
import torch.nn as nn


class InteractionStream(nn.Module):
    """
    Defines a single stream of the IAPE model.
    Contains independent embeddings, a Deep Funnel (MLP), and a Wide Interaction path.
    """

    def __init__(
        self, num_cont, cat_cardinalities, hidden_dims, dropout_rate, emb_dim=16
    ):
        super().__init__()

        # 1. Independent Embeddings
        # Each stream has its own set of embeddings to ensure decorrelated representation learning.
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

        # 3. Wide Path (The Interaction)
        # Input: Continuous features + Pairwise interactions of first 10 embeddings
        # The first 10 categorical features correspond to the characters of f_27.
        # 10 chars -> 10 * 9 / 2 = 45 pairwise interactions.
        self.num_interactions = 45
        wide_input_dim = num_cont + self.num_interactions
        self.wide_linear = nn.Linear(wide_input_dim, 1)

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

        # --- Wide Path ---
        # Use first 10 embeddings for interactions (f_27 characters)
        char_embs = embs[:10]

        interactions = []
        # Compute pairwise dot products (e_i . e_j)
        for i in range(10):
            for j in range(i + 1, 10):
                # Element-wise product then sum over embedding dim
                # (batch, emb_dim) * (batch, emb_dim) -> (batch, emb_dim) -> sum -> (batch, 1)
                dot = (char_embs[i] * char_embs[j]).sum(dim=1, keepdim=True)
                interactions.append(dot)

        interaction_vec = torch.cat(interactions, dim=1)

        # Concatenate continuous features with interaction vector
        wide_in = torch.cat([x_cont, interaction_vec], dim=1)
        wide_out = self.wide_linear(wide_in)

        # --- Aggregation ---
        # Sum the logits from Deep and Wide paths
        return deep_out + wide_out


class IAPEModel(nn.Module):
    """
    Interaction-Augmented Parallel Ensemble (IAPE).
    Consists of multiple independent InteractionStreams instantiated within a single computational graph.
    """

    def __init__(self, num_cont, cat_cardinalities, stream_configs):
        super().__init__()
        self.streams = nn.ModuleList()

        for config in stream_configs:
            stream = InteractionStream(
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
