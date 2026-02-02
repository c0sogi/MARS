import torch
import torch.nn as nn
from library.config import Config


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP architecture.
    Cite solution_lesson_node_00004: Prioritize Funnel MLPs over complex architectures.

    Features:
    - Entity Embeddings for categorical variables.
    - Early fusion of embeddings and continuous features.
    - Funnel-shaped backbone using Linear -> ReLU -> Dropout.
    """

    def __init__(self, vocab_sizes, num_cont_features, cfg=Config):
        super().__init__()

        self.embedding_dim = cfg.EMBEDDING_DIM
        self.dropout_rate = cfg.DROPOUT
        self.hidden_layers = cfg.HIDDEN_LAYERS

        # 1. Entity Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=self.embedding_dim)
                for size in vocab_sizes
            ]
        )

        # 2. Calculate Input Dimension
        total_embedding_dim = len(vocab_sizes) * self.embedding_dim
        input_dim = total_embedding_dim + num_cont_features

        # 3. Backbone: Funnel MLP
        layers = []
        current_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 4. Output Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x_cat, x_cont):
        # Process Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(x_cat[:, i]))

        x_emb = torch.cat(emb_list, dim=1)

        # Early Fusion
        x = torch.cat([x_emb, x_cont], dim=1)

        # Forward pass
        x = self.backbone(x)
        logits = self.head(x)

        return logits
