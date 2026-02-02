import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualBlock(nn.Module):
    """
    A Residual Block with Gated Linear Units (GLU).

    Structure:
        Input -> Linear(d -> 2d) -> GLU(2d -> d) -> Dropout -> Residual Add -> LayerNorm

    This block allows the network to selectively propagate information (gating)
    while maintaining gradient flow via residual connections.
    """

    def __init__(self, input_dim, dropout_rate=0.1):
        """
        Args:
            input_dim (int): The dimensionality of the input and output (residual).
            dropout_rate (float): Probability of dropout.
        """
        super(GatedResidualBlock, self).__init__()
        # GLU requires the input to be split in half, so we project to 2 * input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, input_dim).
        """
        residual = x

        # Project and Gate
        out = self.linear(x)
        out = self.glu(out)
        out = self.dropout(out)

        # Residual Connection and Normalization
        # Post-Norm architecture is used here for stability
        return self.norm(residual + out)


class ECGRN(nn.Module):
    """
    Corrected Entity-Centric Gated Residual Network (EC-GRN-v2).

    This architecture fuses wide-format continuous temporal features with
    entity-specific categorical embeddings. It processes the fused representation
    through a stack of Gated Residual Blocks to capture non-linear interactions
    (e.g., closing speed vs. position type) and outputs raw logits.
    """

    def __init__(
        self,
        num_continuous,
        categorical_embedding_dims,
        hidden_size=512,
        num_blocks=3,
        dropout_rate=0.1,
    ):
        """
        Args:
            num_continuous (int): Number of continuous input features (flattened window).
            categorical_embedding_dims (list of tuples): List of (num_categories, embedding_dim)
                                                         for each categorical feature.
                                                         Order must match the columns in x_cat.
            hidden_size (int): Dimension of the hidden layers in the residual blocks.
            num_blocks (int): Number of GatedResidualBlocks to stack.
            dropout_rate (float): Dropout rate used in blocks.
        """
        super(ECGRN, self).__init__()

        # --- Entity Embeddings ---
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_cats, emb_dim)
                for num_cats, emb_dim in categorical_embedding_dims
            ]
        )

        # Calculate total dimension after concatenation
        total_embedding_dim = sum(emb_dim for _, emb_dim in categorical_embedding_dims)
        input_dim = num_continuous + total_embedding_dim

        # --- Projector ---
        # Projects the mixed input (continuous + embeddings) to the hidden dimension
        self.projector = nn.Linear(input_dim, hidden_size)
        self.projector_norm = nn.LayerNorm(hidden_size)
        self.projector_dropout = nn.Dropout(dropout_rate)

        # --- Backbone ---
        # Stack of Gated Residual Blocks
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_size, dropout_rate) for _ in range(num_blocks)]
        )

        # --- Head ---
        # Single Linear layer to output logits (no activation)
        self.head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Performs Kaiming initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont (torch.Tensor): Continuous features. Shape (Batch, num_continuous).
            x_cat (torch.Tensor): Categorical features (indices). Shape (Batch, num_categorical_features).
                                  The order of columns must match `categorical_embedding_dims`.

        Returns:
            torch.Tensor: Logits. Shape (Batch, 1).
        """
        # 1. Process Embeddings
        embedded_features = []
        # Iterate over the embedding layers and corresponding columns in x_cat
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] selects the column of indices for the i-th categorical feature
            emb = emb_layer(x_cat[:, i])
            embedded_features.append(emb)

        # Concatenate all embeddings along the feature dimension
        if embedded_features:
            x_emb = torch.cat(embedded_features, dim=1)
            # Concatenate continuous features with embeddings
            x = torch.cat([x_cont, x_emb], dim=1)
        else:
            x = x_cont

        # 2. Projection
        x = self.projector(x)
        x = self.projector_norm(x)
        x = self.projector_dropout(x)

        # 3. Backbone (Gated Residual Blocks)
        for block in self.blocks:
            x = block(x)

        # 4. Head (Logits)
        logits = self.head(x)

        return logits
