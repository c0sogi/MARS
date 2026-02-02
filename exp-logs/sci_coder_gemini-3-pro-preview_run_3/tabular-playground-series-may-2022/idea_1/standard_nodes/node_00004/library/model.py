import torch
import torch.nn as nn
from library.config import EMBEDDING_DIM, HIDDEN_LAYERS, DROPOUT_RATE


class ResidualBlock(nn.Module):
    """
    A Residual Block with Linear -> BN -> GELU -> Dropout structure.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class EntityEmbeddingMLP(nn.Module):
    """
    Residual Neural Network with Entity Embeddings for mixed categorical and continuous data.
    """

    def __init__(
        self,
        vocab_sizes,
        num_continuous,
        embedding_dim=EMBEDDING_DIM,
        hidden_layers=HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        Args:
            vocab_sizes (list[int]): A list containing the vocabulary size for each categorical feature.
            num_continuous (int): The number of continuous input features.
            embedding_dim (int): The dimension of the embedding vector.
            hidden_layers (list[int]): A list defining the hidden dimensions.
                                       For ResNet, we use hidden_layers[0] as the block width
                                       and len(hidden_layers) as the number of blocks.
            dropout_rate (float): Dropout probability.
        """
        super(EntityEmbeddingMLP, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.num_continuous = num_continuous

        # Initialize embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embedding_dim)
                for v in vocab_sizes
            ]
        )

        # Calculate input dimension
        input_dim = num_continuous + (len(vocab_sizes) * embedding_dim)

        # Determine ResNet width and depth
        hidden_dim = hidden_layers[0]
        num_blocks = len(hidden_layers)

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous_data, categorical_data):
        # 1. Process Categorical Data
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            col_indices = categorical_data[:, i]
            emb = emb_layer(col_indices)
            embedded_features.append(emb)

        # 2. Combine with Continuous Data
        x = torch.cat(embedded_features + [continuous_data], dim=1)

        # 3. Pass through ResNet
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # 4. Output Prediction
        x = self.output_layer(x)
        output = self.sigmoid(x)

        return output
