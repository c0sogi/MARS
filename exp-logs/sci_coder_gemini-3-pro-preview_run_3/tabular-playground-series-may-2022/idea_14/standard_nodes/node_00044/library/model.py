import torch
import torch.nn as nn
from library.config import Config


class GatedLinearUnit(nn.Module):
    """
    Implements a Gated Linear Unit block consisting of:
    Linear Projection (2x width) -> GLU Activation -> Dropout.

    The GLU operation splits the projected input into two halves (Value and Gate)
    and computes: Value * sigmoid(Gate).
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        # Project to 2x output size to accommodate the split for GLU
        self.linear = nn.Linear(in_features, out_features * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.glu(x)
        x = self.dropout(x)
        return x


class GatedFunnelNetwork(nn.Module):
    """
    Gated Funnel Network architecture.

    Features:
    - Entity Embeddings for categorical variables.
    - Early fusion of embeddings and continuous features.
    - Funnel-shaped backbone using Gated Linear Units (GLU).
    - No Batch/Layer Normalization (relies on Dropout and GLU for stability).
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
        # Flattened embeddings dimension + Continuous features dimension
        total_embedding_dim = len(vocab_sizes) * self.embedding_dim
        input_dim = total_embedding_dim + num_cont_features

        # 3. Backbone: Funnel of Gated Linear Units
        layers = []
        current_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.append(GatedLinearUnit(current_dim, hidden_dim, self.dropout_rate))
            current_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 4. Output Head
        # Direct projection from the last hidden layer to the output logit
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat (torch.Tensor): Categorical features [Batch, Num_Cat_Features]
            x_cont (torch.Tensor): Continuous features [Batch, Num_Cont_Features]
        """
        # Process Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # Lookup embedding for each categorical feature column
            emb_list.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings (Flatten)
        x_emb = torch.cat(emb_list, dim=1)

        # Early Fusion: Combine embeddings and continuous features
        x = torch.cat([x_emb, x_cont], dim=1)

        # Pass through the Gated Funnel backbone
        x = self.backbone(x)

        # Final prediction
        logits = self.head(x)

        return logits
