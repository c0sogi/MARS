import torch
import torch.nn as nn
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron.

    Flattens the temporal window of features into a single wide vector and processes it
    through a dense network. This allows the model to learn time-specific weights for
    each feature in the window (e.g., prioritizing t=0).

    Cite Lesson 00023: Simplicity Enables Data Scale.
    Cite Lesson 00011: Avoids Global Pooling to preserve temporal alignment.
    """

    def __init__(self, config: Config):
        super(KinematicMLP, self).__init__()

        self.window_size = config.window_size
        self.num_features = len(config.feature_cols)
        self.dropout_rate = config.dropout

        # Input size = Window * Features
        self.input_size = self.window_size * self.num_features

        self.model = nn.Sequential(
            nn.Linear(self.input_size, config.dense_hidden_units),
            nn.BatchNorm1d(config.dense_hidden_units),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(config.dense_hidden_units, config.dense_hidden_units),
            nn.BatchNorm1d(config.dense_hidden_units),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(config.dense_hidden_units, config.dense_hidden_units // 2),
            nn.BatchNorm1d(config.dense_hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(config.dense_hidden_units // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Window_Size, Features)
        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        batch_size = x.size(0)

        # Flatten input: (Batch, Window * Features)
        x_flat = x.view(batch_size, -1)

        logits = self.model(x_flat)

        return logits.squeeze(1)
