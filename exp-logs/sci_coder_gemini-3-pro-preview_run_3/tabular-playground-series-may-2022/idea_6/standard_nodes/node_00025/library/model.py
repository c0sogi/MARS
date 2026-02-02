import torch
import torch.nn as nn
from library.config import Config


class HybridTransformerFunnel(nn.Module):
    """
    Hybrid Transformer-Funnel Network Architecture.

    This model processes categorical data as a sequence of interacting tokens using
    a Transformer Encoder, while continuous data is integrated via a Funnel MLP backbone.
    """

    def __init__(self, vocab_sizes, continuous_dim):
        """
        Args:
            vocab_sizes (list or np.ndarray): A list containing the vocabulary size for
                                              each of the 12 categorical features.
            continuous_dim (int): The number of continuous input features.
        """
        super(HybridTransformerFunnel, self).__init__()

        # ----------------------------------------------------------------------
        # Hyperparameters from Config
        # ----------------------------------------------------------------------
        embed_dim = Config.EMBED_DIM
        nhead = Config.TRANSFORMER_HEADS
        num_layers = Config.TRANSFORMER_LAYERS
        dim_feedforward = Config.TRANSFORMER_FF_DIM
        funnel_layers = Config.FUNNEL_LAYERS
        dropout_rate = Config.DROPOUT

        # ----------------------------------------------------------------------
        # 1. Categorical Interaction Branch
        # ----------------------------------------------------------------------
        # Create a specific embedding layer for each position in the sequence
        # because each position corresponds to a distinct feature (e.g., char_0 vs f_29).
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(size), embedding_dim=embed_dim)
                for size in vocab_sizes
            ]
        )

        # Transformer Encoder to capture non-linear dependencies between states/chars
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Calculate the size of the flattened categorical representation
        # Sequence length is fixed at 12 (10 chars + 2 discrete features)
        self.cat_flatten_dim = len(vocab_sizes) * embed_dim

        # ----------------------------------------------------------------------
        # 2. Funnel MLP Backbone (Fusion & Processing)
        # ----------------------------------------------------------------------
        # The MLP input is the concatenation of the flattened transformer output
        # and the continuous feature vector.
        input_dim = self.cat_flatten_dim + continuous_dim

        layers = []
        in_dim = input_dim

        # Build the funnel with decreasing widths (512 -> 256 -> 128)
        for out_dim in funnel_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim

        self.funnel_mlp = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # 3. Classification Head
        # ----------------------------------------------------------------------
        # Projects the final latent representation to a single logit
        self.head = nn.Linear(in_dim, 1)

    def forward(self, cat_seq, cont_vec):
        """
        Forward pass of the network.

        Args:
            cat_seq (torch.Tensor): LongTensor of shape (Batch, 12) containing
                                    indices for categorical features.
            cont_vec (torch.Tensor): FloatTensor of shape (Batch, Continuous_Dim)
                                     containing normalized continuous features.

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # --- Categorical Branch ---
        # Apply specific embedding to each column in the sequence
        # cat_seq[:, i] is (Batch,) -> embedding -> (Batch, Embed_Dim)
        embedded_features = [
            emb(cat_seq[:, i]) for i, emb in enumerate(self.embeddings)
        ]

        # Stack to form the sequence tensor: (Batch, 12, Embed_Dim)
        x_cat = torch.stack(embedded_features, dim=1)

        # Apply Self-Attention via Transformer
        x_cat = self.transformer(x_cat)

        # Flatten the sequence: (Batch, 12 * Embed_Dim)
        x_cat = x_cat.flatten(start_dim=1)

        # --- Fusion ---
        # Concatenate categorical representation with continuous features
        x = torch.cat([x_cat, cont_vec], dim=1)

        # --- Funnel MLP ---
        x = self.funnel_mlp(x)

        # --- Head ---
        logits = self.head(x)

        return logits
