import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Dense Residual Block for WIRK-Net.
    Structure: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Add (Skip Connection)
    """

    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        out += residual
        return out


class WIRKNet(nn.Module):
    """
    Wide-Input Residual Kinematic Network (WIRK-Net).

    A Deep Residual MLP designed for high-dimensional tabular data in contact detection.
    It combines wide-window kinematic features with categorical entity embeddings,
    processed through a deep residual backbone.
    """

    def __init__(
        self,
        num_cont_features,
        cat_vocab_sizes,
        cat_embedding_dims=None,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_RESIDUAL_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            num_cont_features (int): Number of continuous features (flattened wide window).
            cat_vocab_sizes (list[int]): List of vocabulary sizes for each categorical feature.
            cat_embedding_dims (list[int], optional): List of embedding dimensions.
                                                      Defaults to Config.POS_EMBEDDING_DIM for all.
            hidden_dim (int): Dimension of the hidden layers and residual blocks.
            num_blocks (int): Number of residual blocks in the backbone.
            dropout_rate (float): Dropout probability.
        """
        super(WIRKNet, self).__init__()

        # --- Entity Embeddings ---
        if cat_embedding_dims is None:
            # Default to config dimension for all categorical features if not specified
            cat_embedding_dims = [Config.POS_EMBEDDING_DIM] * len(cat_vocab_sizes)

        assert len(cat_vocab_sizes) == len(
            cat_embedding_dims
        ), "Mismatch between vocabulary sizes and embedding dimensions."

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
                for vocab_size, emb_dim in zip(cat_vocab_sizes, cat_embedding_dims)
            ]
        )

        total_emb_dim = sum(cat_embedding_dims)

        # --- Wide Input Projection ---
        # Projects concatenated (Continuous + Embeddings) -> Hidden Dim
        input_dim = num_cont_features + total_emb_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # --- Residual Backbone ---
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout_rate))
        self.backbone = nn.Sequential(*blocks)

        # --- Unified Head ---
        # Outputs logits. Sigmoid is applied during inference or via Loss function (BCEWithLogits).
        self.head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat (torch.Tensor): Categorical features [Batch, Num_Cat_Features] (Long)
            x_cont (torch.Tensor): Continuous features [Batch, Num_Cont_Features] (Float)

        Returns:
            torch.Tensor: Logits [Batch, 1]
        """
        # 1. Process Embeddings
        embedded_features = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] is the i-th categorical feature for the batch
            embedded_features.append(emb_layer(x_cat[:, i]))

        if embedded_features:
            x_emb = torch.cat(embedded_features, dim=1)
            # Concatenate Continuous + Embeddings
            x = torch.cat([x_cont, x_emb], dim=1)
        else:
            x = x_cont

        # 2. Projection
        x = self.input_proj(x)

        # 3. Residual Backbone
        x = self.backbone(x)

        # 4. Head
        logits = self.head(x)

        return logits
