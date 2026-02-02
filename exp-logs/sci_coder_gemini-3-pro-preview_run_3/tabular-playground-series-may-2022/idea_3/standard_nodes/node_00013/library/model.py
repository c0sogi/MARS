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


class Decoder(nn.Module):
    """
    Decoder network for the Denoising Autoencoder.
    Mirrors the Encoder to reconstruct inputs.
    """

    def __init__(self, input_dim, cont_dim, cat_cardinalities):
        super().__init__()

        # ---------------------------------------------------------
        # Expanding MLP
        # ---------------------------------------------------------
        layers = []
        in_dim = input_dim

        for out_dim in Config.DECODER_LAYERS:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # ---------------------------------------------------------
        # Reconstruction Heads
        # ---------------------------------------------------------
        # 1. Continuous feature reconstruction (Linear output for MSE)
        self.cont_head = nn.Linear(in_dim, cont_dim)

        # 2. Categorical feature reconstruction (Linear output for CrossEntropy logits)
        self.cat_heads = nn.ModuleList()
        for card in cat_cardinalities:
            self.cat_heads.append(nn.Linear(in_dim, card))

    def forward(self, z):
        """
        Args:
            z: Latent representation from Encoder (batch_size, latent_dim)
        Returns:
            rec_cont: Reconstructed continuous features
            rec_cats: List of tensors containing logits for each categorical feature
        """
        h = self.mlp(z)

        rec_cont = self.cont_head(h)

        rec_cats = []
        for head in self.cat_heads:
            rec_cats.append(head(h))

        return rec_cont, rec_cats


class DenoisingAutoencoder(nn.Module):
    """
    Composite model combining Encoder and Decoder for unsupervised pretraining.
    """

    def __init__(self, cont_dim, cat_cardinalities):
        super().__init__()
        self.encoder = Encoder(cont_dim, cat_cardinalities)
        self.decoder = Decoder(self.encoder.output_dim, cont_dim, cat_cardinalities)

    def forward(self, x_cont, x_cat):
        """
        Forward pass through Encoder and Decoder.
        """
        z = self.encoder(x_cont, x_cat)
        return self.decoder(z)


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
