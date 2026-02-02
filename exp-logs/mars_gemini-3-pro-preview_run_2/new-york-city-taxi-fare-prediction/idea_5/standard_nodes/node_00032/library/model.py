import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A standard residual block for dense neural networks.
    Structure: Input -> [Linear->BN->ReLU->Dropout->Linear->BN] + Input -> ReLU
    """

    def __init__(self, dim, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


class SpatialResNet(nn.Module):
    """
    Deep Residual MLP with Spatial Embeddings for Taxi Fare Prediction.

    Architecture:
    1. Embeddings for spatial grids and temporal features.
    2. Concatenation of embeddings with continuous physical features.
    3. Projection to hidden dimension.
    4. Stack of Residual Blocks.
    5. Final regression head.
    """

    def __init__(
        self,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
        grid_bins=Config.GRID_BINS,
    ):
        super(SpatialResNet, self).__init__()

        # 1. Define Embedding Layers
        # The order corresponds to DataProcessor.categorical_cols:
        # [pickup_grid_lat, pickup_grid_lon, dropoff_grid_lat, dropoff_grid_lon,
        #  hour, weekday, day, month, year]

        # Define vocabulary sizes based on data ranges
        self.vocab_sizes = [
            grid_bins,  # pickup_grid_lat (0 to bins-1)
            grid_bins,  # pickup_grid_lon (0 to bins-1)
            grid_bins,  # dropoff_grid_lat (0 to bins-1)
            grid_bins,  # dropoff_grid_lon (0 to bins-1)
            24,  # hour (0-23)
            7,  # weekday (0-6)
            32,  # day (1-31)
            13,  # month (1-12)
            10,  # year (0-6 approx, 10 is safe upper bound)
        ]

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vocab, embedding_dim=embedding_dim)
                for vocab in self.vocab_sizes
            ]
        )

        # 2. Calculate Input Dimension
        # Continuous features count is 11 (as defined in DataProcessor.continuous_cols)
        self.num_continuous = 11
        self.total_input_dim = self.num_continuous + (
            len(self.vocab_sizes) * embedding_dim
        )

        # 3. Input Projection Layer
        self.input_proj = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # 4. Residual Backbone
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )

        # 5. Output Head (Scalar Prediction)
        self.head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, continuous_features, categorical_indices):
        """
        Forward pass of the network.

        Args:
            continuous_features (torch.Tensor): Continuous input features of shape (batch_size, 11).
            categorical_indices (torch.Tensor): Categorical indices of shape (batch_size, 9).

        Returns:
            torch.Tensor: Predicted fare amount of shape (batch_size, 1).
        """
        # 1. Process Embeddings
        embedded_list = []
        for i, layer in enumerate(self.embeddings):
            # Extract column i from categorical inputs
            idx = categorical_indices[:, i]
            # Get embedding vectors
            embedded_list.append(layer(idx))

        # Concatenate all embeddings: (batch, num_cats * emb_dim)
        x_emb = torch.cat(embedded_list, dim=1)

        # 2. Concatenate with Continuous Features
        # Result shape: (batch, total_input_dim)
        x = torch.cat([continuous_features, x_emb], dim=1)

        # 3. Project to Hidden Dimension
        x = self.input_proj(x)

        # 4. Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)

        # 5. Output Prediction
        out = self.head(x)

        return out
