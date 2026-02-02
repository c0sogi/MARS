import torch
import torch.nn as nn
from library.config import Config


class CrossLayer(nn.Module):
    """
    Cross Layer as defined in Deep & Cross Network (DCN).
    Implements the vector-wise interaction formula:
    x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l

    This explicitly models feature interactions of bounded degree.
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim

        # Weight vector w_l implemented as a Linear layer mapping input_dim -> 1.
        # This computes the scalar projection (x_l^T * w_l) for each sample in the batch.
        self.linear = nn.Linear(input_dim, 1, bias=False)

        # Bias vector b_l
        self.bias = nn.Parameter(torch.zeros(input_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x0, xl):
        """
        Args:
            x0 (torch.Tensor): Initial input features of shape (Batch, Input_Dim).
            xl (torch.Tensor): Output from the previous layer of shape (Batch, Input_Dim).

        Returns:
            torch.Tensor: The output of the cross layer x_{l+1}.
        """
        # Compute interaction score (scalar per sample): x_l^T * w_l
        # Shape: (Batch, 1)
        score = self.linear(xl)

        # Apply formula: x_0 * score + b + x_l
        # Broadcasting handles the scalar multiplication and bias addition
        out = x0 * score + self.bias + xl
        return out


class ResNetBlock(nn.Module):
    """
    Residual Block for the Deep Network component.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResNetBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.layer(x)


class DCNV2(nn.Module):
    """
    Deep & Cross Network V2 Architecture.

    Combines:
    1. Entity Embeddings for high-cardinality categorical features.
    2. A Cross Network for explicit, bounded-degree feature interactions.
    3. A Deep ResNet-MLP for implicit, high-order non-linear patterns.
    """

    def __init__(self, num_cont_features):
        """
        Args:
            num_cont_features (int): The number of continuous input features.
        """
        super(DCNV2, self).__init__()

        # ---------------------------------------------------------
        # 1. Embeddings
        # ---------------------------------------------------------
        # Soil Type Embedding
        self.soil_emb = nn.Embedding(
            num_embeddings=Config.EMBEDDING_CONFIG["soil"]["num_embeddings"],
            embedding_dim=Config.EMBEDDING_CONFIG["soil"]["embedding_dim"],
        )
        # Wilderness Area Embedding
        self.wild_emb = nn.Embedding(
            num_embeddings=Config.EMBEDDING_CONFIG["wilderness"]["num_embeddings"],
            embedding_dim=Config.EMBEDDING_CONFIG["wilderness"]["embedding_dim"],
        )

        total_emb_dim = (
            Config.EMBEDDING_CONFIG["soil"]["embedding_dim"]
            + Config.EMBEDDING_CONFIG["wilderness"]["embedding_dim"]
        )

        # Total dense input dimension D = Continuous Features + Embedding Dimensions
        self.input_dim = num_cont_features + total_emb_dim

        # ---------------------------------------------------------
        # 2. Cross Network
        # ---------------------------------------------------------
        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.input_dim) for _ in range(Config.NUM_CROSS_LAYERS)]
        )

        # ---------------------------------------------------------
        # 3. Deep Network (ResNet Backbone)
        # ---------------------------------------------------------
        # Initial projection from input dimension to hidden dimension
        hidden_dim = Config.HIDDEN_UNITS[0]
        self.deep_input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.deep_act = nn.ReLU()

        # Construct Deep Blocks
        deep_blocks = []
        current_dim = hidden_dim

        for h_dim in Config.HIDDEN_UNITS:
            # If the hidden size changes between blocks (though Config suggests constant [256, 256]),
            # we add a linear projection to match dimensions.
            if h_dim != current_dim:
                deep_blocks.append(nn.Linear(current_dim, h_dim))
                deep_blocks.append(nn.ReLU())
                current_dim = h_dim

            # Add Residual Block
            deep_blocks.append(ResNetBlock(current_dim, Config.DROPOUT_RATE))

        self.deep_network = nn.Sequential(*deep_blocks)
        self.deep_out_dim = current_dim

        # ---------------------------------------------------------
        # 4. Output Layer
        # ---------------------------------------------------------
        # The final representation is the concatenation of the Cross Network output (Input Dim)
        # and the Deep Network output (Hidden Dim).
        stack_dim = self.input_dim + self.deep_out_dim
        self.output_layer = nn.Linear(stack_dim, Config.NUM_CLASSES)

    def forward(self, x_cont, x_cat):
        """
        Forward pass of the DCN-V2 model.

        Args:
            x_cont (torch.Tensor): Continuous features of shape (Batch, Num_Cont).
            x_cat (torch.Tensor): Categorical indices of shape (Batch, 2).
                                  x_cat[:, 0] -> Soil Type Index
                                  x_cat[:, 1] -> Wilderness Area Index

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # 1. Process Embeddings
        soil_emb = self.soil_emb(x_cat[:, 0])
        wild_emb = self.wild_emb(x_cat[:, 1])

        # Concatenate continuous features and embeddings to form x0
        x0 = torch.cat([x_cont, soil_emb, wild_emb], dim=1)

        # 2. Cross Network Forward Pass
        # x_{l+1} depends on x_l and x_0
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        cross_out = xl

        # 3. Deep Network Forward Pass
        deep_out = self.deep_input_proj(x0)
        deep_out = self.deep_act(deep_out)
        deep_out = self.deep_network(deep_out)

        # 4. Combination & Output
        # Stack features from both networks
        stacked = torch.cat([cross_out, deep_out], dim=1)
        logits = self.output_layer(stacked)

        return logits
