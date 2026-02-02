import torch
import torch.nn as nn
import timm
from library.config import Config


class FiLMGenerator(nn.Module):
    """
    A Multi-Layer Perceptron that generates Feature-wise Linear Modulation (FiLM) parameters
    (gamma and beta) from a vector of global signal scalars.

    Structure: BatchNorm -> Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FiLMGenerator, self).__init__()

        # Normalize scalar inputs within the model flow
        self.bn = nn.BatchNorm1d(input_dim)

        # MLP to project scalars to modulation parameters
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, output_dim * 2
            ),  # Outputs concatenated gamma and beta
        )

        self._init_weights(output_dim)

    def _init_weights(self, output_dim):
        """
        Initializes the last layer to approximate an identity mapping at the start of training.
        Gamma (scale) is initialized to 1.
        Beta (shift) is initialized to 0.
        """
        # Access the last Linear layer in the Sequential block
        last_layer = self.mlp[2]

        # Initialize weights to be very small
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)

        # Initialize bias: first half (gamma) to 1, second half (beta) to 0
        nn.init.constant_(last_layer.bias[:output_dim], 1.0)
        nn.init.constant_(last_layer.bias[output_dim:], 0.0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Scalar input vector (Batch, input_dim)

        Returns:
            tuple: (gamma, beta), each of shape (Batch, output_dim)
        """
        x = self.bn(x)
        out = self.mlp(x)

        # Split output into scale (gamma) and shift (beta)
        gamma, beta = torch.chunk(out, 2, dim=1)
        return gamma, beta


class EfficientNetFiLM(nn.Module):
    """
    EfficientNet-B0 architecture enhanced with Feature-wise Linear Modulation.

    This model fuses high-dimensional texture features from spectrograms with
    explicit magnitude information from global scalars.
    """

    def __init__(self, scalar_input_dim):
        super(EfficientNetFiLM, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # in_chans=10: Adapts the first conv layer to handle 10 sensor channels.
        # num_classes=0: Returns the global pooled feature vector (1280 dim).
        self.backbone = timm.create_model(
            Config.EFFICIENTNET_VERSION,
            pretrained=True,
            in_chans=Config.NUM_SENSORS,
            num_classes=0,
        )

        # Determine feature dimension (1280 for EfficientNet-B0)
        self.feature_dim = self.backbone.num_features

        # 2. FiLM Generator
        # Projects scalars to the feature dimension of the backbone
        self.film_generator = FiLMGenerator(
            input_dim=scalar_input_dim,
            hidden_dim=Config.FILM_HIDDEN_DIM,
            output_dim=self.feature_dim,
        )

        # 3. Regression Head
        self.regressor = nn.Linear(self.feature_dim, 1)

    def forward(self, x_spec, x_scalar):
        """
        Forward pass of the Magnitude-Modulated Vision Model.

        Args:
            x_spec (torch.Tensor): Spectrogram input (Batch, 10, 256, 256).
            x_scalar (torch.Tensor): Global scalar stats (Batch, scalar_input_dim).

        Returns:
            torch.Tensor: Predicted log(time_to_eruption + 1) (Batch, 1).
        """
        # Extract visual features from spectrograms
        # Shape: (Batch, 1280)
        features = self.backbone(x_spec)

        # Generate modulation parameters from scalars
        # Shapes: (Batch, 1280)
        gamma, beta = self.film_generator(x_scalar)

        # Apply Feature-wise Linear Modulation
        # E_mod = gamma * E_cnn + beta
        # This allows magnitude info to scale/shift the texture features
        modulated_features = features * gamma + beta

        # Predict target
        output = self.regressor(modulated_features)

        return output
