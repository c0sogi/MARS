import torch
import torch.nn as nn
from library.config import Config


class CrossLayer(nn.Module):
    """
    Implements a single layer of the Cross Network branch.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    This explicitly models feature interactions of degree l+1.
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim

        # Learnable parameters: weight vector w and bias vector b
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        # Initialize parameters
        # Use a small uniform distribution for weights to start with weak interactions
        nn.init.uniform_(self.weight, -0.05, 0.05)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (Batch, D)
            xl: Output from the previous layer (Batch, D)
        Returns:
            x_next: (Batch, D)
        """
        # Compute scalar score (x_l . w) for each sample in the batch
        # xl: (B, D), weight: (D,) -> element-wise mul -> sum over D -> (B, 1)
        score = torch.sum(xl * self.weight, dim=1, keepdim=True)

        # Multiply x0 by the scalar score (broadcasting)
        crossed = x0 * score

        # Add bias and residual connection
        output = crossed + self.bias + xl
        return output


class FunnelMLP(nn.Module):
    """
    Implements the Deep Branch using a Funnel MLP architecture.
    Layers decrease in width (e.g., 512 -> 256 -> 128) to compress features.
    """

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(FunnelMLP, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.network(x)


class DCNv2(nn.Module):
    """
    Deep Cross Network V2 (DCNv2) architecture.
    Combines a Cross Network branch for explicit feature interactions and
    a Deep Network branch (Funnel MLP) for implicit non-linear representations.
    """

    def __init__(self):
        super(DCNv2, self).__init__()

        # --- Input Dimensions ---
        # Continuous features: Base features (30) + 1 engineered feature (unique_char_count)
        self.num_cont = len(Config.BASE_CONT_COLS) + 1

        # Categorical features: 10 character positions from f_27
        self.num_cat = Config.STR_LEN
        self.emb_dim = Config.EMBEDDING_DIM
        self.vocab_size = Config.VOCAB_SIZE

        # Total flattened input dimension D
        # D = num_cont + (num_cat * emb_dim)
        self.total_input_dim = self.num_cont + (self.num_cat * self.emb_dim)

        # --- Components ---

        # 1. Entity Embeddings for characters
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)

        # 2. Cross Network Branch
        # Stack of CrossLayers
        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.total_input_dim) for _ in range(Config.NUM_CROSS_LAYERS)]
        )

        # 3. Deep Network Branch (Funnel MLP)
        self.deep_net = FunnelMLP(
            input_dim=self.total_input_dim,
            hidden_layers=Config.HIDDEN_LAYERS,
            dropout_rate=Config.DROPOUT,
        )

        # 4. Final Combination Layer
        # Concatenate outputs of Cross (D) and Deep (Last Hidden)
        final_dim = self.total_input_dim + self.deep_net.output_dim
        self.output_layer = nn.Linear(final_dim, 1)

    def forward(self, continuous, categorical):
        """
        Forward pass of the DCNv2 model.

        Args:
            continuous: Tensor of shape (Batch, num_cont)
            categorical: Tensor of shape (Batch, num_cat) containing integer indices

        Returns:
            probs: Tensor of shape (Batch, 1) containing predicted probabilities
        """
        # 1. Prepare Input Vector x0
        # Lookup embeddings: (Batch, 10) -> (Batch, 10, 8)
        emb = self.embeddings(categorical)
        # Flatten embeddings: (Batch, 80)
        emb_flat = emb.view(emb.size(0), -1)

        # Concatenate continuous features and flattened embeddings
        x0 = torch.cat([continuous, emb_flat], dim=1)

        # 2. Cross Branch
        # Pass x0 through the stack of CrossLayers
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        x_cross = xl

        # 3. Deep Branch
        # Pass x0 through the Funnel MLP
        x_deep = self.deep_net(x0)

        # 4. Combination
        # Concatenate the outputs from both branches
        x_concat = torch.cat([x_cross, x_deep], dim=1)

        # 5. Output
        # Linear projection to 1 unit -> Sigmoid for probability
        logits = self.output_layer(x_concat)
        probs = torch.sigmoid(logits)

        return probs
