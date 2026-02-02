import torch
import torch.nn as nn
from library.config import Config


class TreeFunnelEnsemble(nn.Module):
    """
    Tree-Structured Funnel Ensemble model.

    Architecture:
    1. Inputs: Categorical Entity Embeddings + Normalized Continuous Features.
    2. Trunk: Shared Dense Block (Linear -> ReLU -> Dropout) learning common representations.
    3. Heads: Multiple independent branches (Funnel MLPs) predicting the target.

    This structure internalizes ensembling and allows for diverse feature extraction
    from a shared low-level representation.
    """

    def __init__(self, vocab_sizes, cont_dim):
        """
        Args:
            vocab_sizes (list[int]): List containing the vocabulary size for each categorical feature.
            cont_dim (int): Number of continuous features.
        """
        super(TreeFunnelEnsemble, self).__init__()

        # Hyperparameters from Config
        self.num_heads = Config.NUM_HEADS
        embed_dim = Config.EMBED_DIM
        trunk_dim = Config.TRUNK_HIDDEN_DIM
        head_dims = Config.HEAD_HIDDEN_DIMS
        dropout_rate = Config.DROPOUT_RATE

        # 1. Entity Embeddings
        # Create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate total input dimension for the trunk
        # Input = (Num Categorical * Embed Dim) + Num Continuous
        cat_input_dim = len(vocab_sizes) * embed_dim
        total_input_dim = cat_input_dim + cont_dim

        # 2. Shared Trunk (The Root)
        # Extracts fundamental, shared features.
        # Structure: Linear(Input -> 512) -> ReLU -> Dropout
        self.trunk = nn.Sequential(
            nn.Linear(total_input_dim, trunk_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # 3. Independent Funnel Heads (The Branches)
        # Each head learns a complete predictive function from the shared features.
        # Structure defined by HEAD_HIDDEN_DIMS (e.g., 512 -> 256 -> 128 -> 1)
        self.heads = nn.ModuleList()

        for _ in range(self.num_heads):
            layers = []
            input_dim = trunk_dim

            # Build hidden layers for the funnel
            for hidden_dim in head_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                input_dim = hidden_dim

            # Final projection to logits (Linear -> 1)
            # No Sigmoid here; BCEWithLogitsLoss will be used during training.
            layers.append(nn.Linear(input_dim, 1))

            self.heads.append(nn.Sequential(*layers))

    def forward(self, cat_x, cont_x):
        """
        Forward pass of the model.

        Args:
            cat_x (torch.LongTensor): Categorical features of shape (batch_size, num_cat_features).
            cont_x (torch.FloatTensor): Continuous features of shape (batch_size, num_cont_features).

        Returns:
            list[torch.Tensor]: A list of tensors, where each tensor is the output (logits)
                                from one head with shape (batch_size, 1).
        """
        # 1. Process Embeddings
        # Lookup embeddings for each categorical feature
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            # cat_x[:, i] is the i-th categorical feature column
            embeds.append(emb_layer(cat_x[:, i]))

        # Concatenate all embeddings along the feature dimension
        # Shape: (batch_size, num_cat_features * embed_dim)
        cat_features = torch.cat(embeds, dim=1)

        # 2. Combine Features
        # Concatenate embeddings with continuous features
        # Shape: (batch_size, total_input_dim)
        x = torch.cat([cat_features, cont_x], dim=1)

        # 3. Pass through Shared Trunk
        # Shape: (batch_size, trunk_dim)
        trunk_out = self.trunk(x)

        # 4. Pass through Independent Heads
        # Collect outputs from all branches
        outputs = []
        for head in self.heads:
            outputs.append(head(trunk_out))

        return outputs
