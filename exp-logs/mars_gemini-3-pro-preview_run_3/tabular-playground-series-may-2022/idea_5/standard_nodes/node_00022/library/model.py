import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedBlock(nn.Module):
    """
    A Transformer-style Gated Block adapted for a funnel topology.
    Structure: LayerNorm -> Linear (Expand 2x) -> GLU -> Dropout.
    """

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        # GLU requires input size 2 * out_dim to produce output size out_dim
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        return x


class LNGatedFunnelNet(nn.Module):
    """
    Layer-Normalized Gated Funnel Network.
    Combines Entity Embeddings for categorical data and a Gated MLP backbone
    with Layer Normalization for stable convergence on tabular data.
    """

    def __init__(
        self, num_cont, cat_cardinalities, embedding_dim, hidden_layers, dropout
    ):
        super().__init__()

        # Entity Embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )

        # Calculate input dimension: Continuous features + Flattened embeddings
        # Total embedding size = number of categorical features * embedding_dim
        self.input_dim = num_cont + len(cat_cardinalities) * embedding_dim

        # Build the backbone (Funnel structure)
        layers = []
        in_dim = self.input_dim

        for h_dim in hidden_layers:
            layers.append(GatedBlock(in_dim, h_dim, dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Output Head
        # Direct connection from the last hidden representation to the output neuron
        self.head = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: Tensor of shape [batch_size, num_cont]
            x_cat: LongTensor of shape [batch_size, num_cat]
        """
        # Process Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select the i-th categorical column
            emb_list.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings: [batch, num_cat * emb_dim]
        x_emb = torch.cat(emb_list, dim=1)

        # Concatenate continuous features with embeddings
        # [batch, num_cont + num_cat * emb_dim]
        x = torch.cat([x_cont, x_emb], dim=1)

        # Pass through the Gated Funnel Backbone
        x = self.backbone(x)

        # Final classification head
        return self.head(x)
