import torch
import torch.nn as nn
from library.config import Config


class Encoder(nn.Module):
    """
    Encoder backbone using a Funnel MLP architecture.
    Accepts continuous features and categorical inputs (via embeddings).
    """

    def __init__(self, cont_dim, cat_cardinalities):
        super().__init__()

        # ---------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------
        self.embeddings = nn.ModuleList()
        cat_dims = []

        # The order in cat_cardinalities corresponds to:
        # [f_27_0, ..., f_27_9, f_29, f_30]
        # First F27_SEQ_LEN are characters, rest are discrete
        num_chars = Config.F27_SEQ_LEN

        for i, card in enumerate(cat_cardinalities):
            if i < num_chars:
                emb_dim = Config.CHAR_EMBEDDING_DIM
            else:
                emb_dim = Config.DISCRETE_EMBEDDING_DIM

            self.embeddings.append(
                nn.Embedding(num_embeddings=card, embedding_dim=emb_dim)
            )
            cat_dims.append(emb_dim)

        # Total input dimension = Continuous features + Concatenated Embeddings
        self.input_dim = cont_dim + sum(cat_dims)

        # ---------------------------------------------------------
        # Funnel MLP
        # ---------------------------------------------------------
        layers = []
        in_dim = self.input_dim

        for out_dim in Config.ENCODER_LAYERS:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = Config.ENCODER_LAYERS[-1]

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: (batch_size, cont_dim)
            x_cat: (batch_size, num_cat_features)
        Returns:
            z: Latent representation (batch_size, output_dim)
        """
        # Embed categorical features
        embedded_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] shape is (batch_size,)
            embedded_list.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings: (batch_size, total_emb_dim)
        x_emb = torch.cat(embedded_list, dim=1)

        # Concatenate with continuous features: (batch_size, input_dim)
        x = torch.cat([x_cont, x_emb], dim=1)

        # Pass through MLP
        z = self.mlp(x)
        return z


class ManufacturingClassifier(nn.Module):
    """
    Classifier model using the pretrained Encoder backbone.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Classification Head
        # Simple dense layer mapping latent dim to 1 output logit
        # Sigmoid activation is omitted here to allow usage of BCEWithLogitsLoss
        self.head = nn.Linear(encoder.output_dim, 1)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: Continuous features
            x_cat: Categorical features
        Returns:
            logits: Unnormalized logits (batch_size, 1)
        """
        z = self.encoder(x_cont, x_cat)
        logits = self.head(z)
        return logits
