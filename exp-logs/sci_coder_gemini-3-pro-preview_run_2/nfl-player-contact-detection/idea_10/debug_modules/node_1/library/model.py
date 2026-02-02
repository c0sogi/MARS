import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A dense residual block for the EF-WideResNet.
    Structure: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )  # Adding BN2 for stability before addition

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)  # Final activation after residual connection
        return out


class EFWideResNet(nn.Module):
    """
    Explicitly-Featurized Wide Residual Network.

    A Deep Residual MLP designed for high-dimensional, explicitly engineered tabular data.
    It combines entity embeddings for categorical features with a wide continuous feature vector,
    processed through a stack of residual blocks.
    """

    def __init__(
        self,
        num_continuous_features,
        embedding_config=Config.EMBEDDING_CONFIG,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            num_continuous_features (int): Dimension of the flattened continuous feature vector.
            embedding_config (dict): Configuration for embedding layers (from Config).
            hidden_dim (int): Width of the hidden layers and residual blocks.
            num_res_blocks (int): Number of residual blocks to stack.
            dropout_rate (float): Dropout probability.
        """
        super(EFWideResNet, self).__init__()

        self.embedding_config = embedding_config
        self.embeddings = nn.ModuleDict()

        total_embedding_dim = 0
        for name, cfg in embedding_config.items():
            self.embeddings[name] = nn.Embedding(
                num_embeddings=cfg["num_embeddings"], embedding_dim=cfg["embedding_dim"]
            )
            total_embedding_dim += cfg["embedding_dim"]

        # Calculate total input dimension
        input_dim = num_continuous_features + total_embedding_dim

        # Input Projection: Map wide input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Residual Stack
        layers = []
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout_rate))
        self.res_blocks = nn.Sequential(*layers)

        # Output Head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, continuous_feats, categorical_feats):
        """
        Args:
            continuous_feats (torch.Tensor): Tensor of shape (batch_size, num_continuous_features).
            categorical_feats (dict[str, torch.Tensor]): Dictionary mapping feature names to
                                                         tensors of shape (batch_size,).

        Returns:
            torch.Tensor: Logits of shape (batch_size, 1).
        """
        # Process Embeddings
        embedded_list = []
        for name, layer in self.embeddings.items():
            if name in categorical_feats:
                # Ensure input is long tensor for embedding lookup
                idx = categorical_feats[name].long()
                embedded = layer(idx)
                embedded_list.append(embedded)
            else:
                raise ValueError(f"Missing categorical feature input: {name}")

        # Concatenate continuous features and embeddings
        if embedded_list:
            x_cat = torch.cat(embedded_list, dim=1)
            x = torch.cat([continuous_feats, x_cat], dim=1)
        else:
            x = continuous_feats

        # Project to hidden dimension
        x = self.input_proj(x)

        # Pass through residual blocks
        x = self.res_blocks(x)

        # Final prediction (Logits)
        out = self.head(x)

        return out
