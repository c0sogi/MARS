import torch
import torch.nn as nn
from library.config import Config


class FunnelMLP(nn.Module):
    """
    Standard Funnel MLP with Entity Embeddings.
    Cite solution_lesson_node_00047: Reverting to standard MLP outperformed complex architectures.

    Architecture:
    1. Entity Embeddings for categorical variables.
    2. Concatenation of Continuous features and Flattened Embeddings.
    3. Funnel MLP backbone with ReLU activations and Dropout.
    """

    def __init__(
        self,
        num_continuous,
        categorical_vocab_sizes,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        output_dim=Config.OUTPUT_DIM,
    ):
        super().__init__()

        # 1. Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
                for vocab_size in categorical_vocab_sizes
            ]
        )

        # Calculate total input dimension for the backbone
        total_input_dim = num_continuous + (
            len(categorical_vocab_sizes) * embedding_dim
        )

        # 2. Funnel Backbone
        layers = []
        in_dim = total_input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Output Head
        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, x_cont, x_cat):
        # Process Embeddings
        embedded_list = []
        for i, emb_layer in enumerate(self.embeddings):
            col_indices = x_cat[:, i]
            embedded_list.append(emb_layer(col_indices))

        x_emb = torch.cat(embedded_list, dim=1)

        # Concatenate with continuous features
        x = torch.cat([x_cont, x_emb], dim=1)

        # Pass through Backbone
        x = self.backbone(x)

        # Output Head
        logits = self.head(x)

        return logits
