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


class ManufacturingMLP(nn.Module):
    """
    Standard Funnel MLP with Entity Embeddings for both characters and discrete features.
    Replaces DCNv2 to avoid overfitting (Cite solution_lesson_node_00006).
    """

    def __init__(self):
        super(ManufacturingMLP, self).__init__()

        # --- Input Dimensions ---
        # Continuous features: Base features (28) + 1 engineered feature (unique_char_count)
        self.num_cont = len(Config.BASE_CONT_COLS) + 1

        # Character Embedding Config
        self.char_emb_dim = Config.EMBEDDING_DIM
        self.vocab_size = Config.VOCAB_SIZE

        # Discrete Feature Embeddings
        # f_29: 2 values -> dim 4
        # f_30: 3 values -> dim 4
        self.f29_emb_dim = 4
        self.f30_emb_dim = 4

        # Total flattened input dimension
        # Continuous (29) + Chars (10*8=80) + f29 (4) + f30 (4) = 117
        self.total_input_dim = (
            self.num_cont
            + (Config.STR_LEN * self.char_emb_dim)
            + self.f29_emb_dim
            + self.f30_emb_dim
        )

        # --- Components ---

        # 1. Entity Embeddings
        self.char_embeddings = nn.Embedding(self.vocab_size, self.char_emb_dim)
        self.f29_embedding = nn.Embedding(2, self.f29_emb_dim)
        self.f30_embedding = nn.Embedding(3, self.f30_emb_dim)

        # 2. Deep Network Branch (Funnel MLP)
        self.deep_net = FunnelMLP(
            input_dim=self.total_input_dim,
            hidden_layers=Config.HIDDEN_LAYERS,
            dropout_rate=Config.DROPOUT,
        )

        # 3. Output Layer
        self.output_layer = nn.Linear(self.deep_net.output_dim, 1)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: (Batch, num_cont)
            categorical: (Batch, 12) -> [char_0...char_9, f_29, f_30]
        """
        # Split categorical inputs
        chars = categorical[:, :10]  # First 10 cols are characters
        f29 = categorical[:, 10]  # 11th col is f_29
        f30 = categorical[:, 11]  # 12th col is f_30

        # Embed
        # Chars: (Batch, 10) -> (Batch, 10, 8) -> (Batch, 80)
        emb_chars = self.char_embeddings(chars).view(chars.size(0), -1)

        # Discrete: (Batch,) -> (Batch, 4)
        emb_f29 = self.f29_embedding(f29)
        emb_f30 = self.f30_embedding(f30)

        # Concatenate all features
        x = torch.cat([continuous, emb_chars, emb_f29, emb_f30], dim=1)

        # Pass through MLP
        x = self.deep_net(x)

        # Output
        logits = self.output_layer(x)
        probs = torch.sigmoid(logits)

        return probs
