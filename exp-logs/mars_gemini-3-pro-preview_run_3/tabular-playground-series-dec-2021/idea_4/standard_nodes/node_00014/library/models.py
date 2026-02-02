import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A standard Residual Block for tabular data.
    Structure: Input -> [BN -> ReLU -> Dropout -> Linear] x 2 -> Add Input
    """

    def __init__(self, hidden_dim, dropout=Config.DROPOUT):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    Encoder part of the DAE. Maps raw features to a latent representation.
    """

    def __init__(
        self, input_dim, hidden_dim=Config.HIDDEN_DIM, latent_dim=Config.LATENT_DIM
    ):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder part of the DAE. Reconstructs features from the latent representation.
    """

    def __init__(
        self,
        latent_dim=Config.LATENT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=None,
    ):
        super(Decoder, self).__init__()
        if output_dim is None:
            raise ValueError("output_dim must be specified for Decoder")

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder (DAE) combining Encoder and Decoder.
    """

    def __init__(
        self, input_dim, hidden_dim=Config.HIDDEN_DIM, latent_dim=Config.LATENT_DIM
    ):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim=input_dim)

    def forward(self, x):
        # x should be the noisy input during training
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class ResNetClassifier(nn.Module):
    """
    ResNet-MLP Classifier that sits on top of the Encoder.
    """

    def __init__(
        self,
        encoder,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        num_blocks=2,
    ):
        super(ResNetClassifier, self).__init__()
        self.encoder = encoder

        # Project latent dim to hidden dim for ResNet blocks
        self.input_proj = nn.Linear(Config.LATENT_DIM, hidden_dim)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Pass through encoder to get latent representation
        z = self.encoder(x)

        # Project and pass through ResNet blocks
        h = self.input_proj(z)
        h = self.blocks(h)

        # Classification head
        logits = self.head(h)
        return logits
