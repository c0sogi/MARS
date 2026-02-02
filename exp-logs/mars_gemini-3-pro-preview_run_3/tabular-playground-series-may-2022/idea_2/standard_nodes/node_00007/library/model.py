import torch
import torch.nn as nn
from library.config import Config


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


class TabularMLP(nn.Module):
    """
    Standard Funnel MLP architecture with Entity Embeddings.
    Cite Lesson 00006: Prefer standard MLPs over specialized interaction architectures (DCN)
    for this dataset to avoid overfitting.
    """

    def __init__(self):
        super(TabularMLP, self).__init__()

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

        # 2. Deep Network Branch (Funnel MLP)
        self.deep_net = FunnelMLP(
            input_dim=self.total_input_dim,
            hidden_layers=Config.HIDDEN_LAYERS,
            dropout_rate=Config.DROPOUT,
        )

        # 3. Output Layer
        # Linear projection from last hidden layer to 1 unit
        self.output_layer = nn.Linear(self.deep_net.output_dim, 1)

    def forward(self, continuous, categorical):
        """
        Forward pass of the TabularMLP model.
        """
        # 1. Prepare Input Vector
        # Lookup embeddings: (Batch, 10) -> (Batch, 10, 8)
        emb = self.embeddings(categorical)
        # Flatten embeddings: (Batch, 80)
        emb_flat = emb.view(emb.size(0), -1)

        # Concatenate continuous features and flattened embeddings
        x = torch.cat([continuous, emb_flat], dim=1)

        # 2. Deep Network
        x_deep = self.deep_net(x)

        # 3. Output
        logits = self.output_layer(x_deep)
        probs = torch.sigmoid(logits)

        return probs
