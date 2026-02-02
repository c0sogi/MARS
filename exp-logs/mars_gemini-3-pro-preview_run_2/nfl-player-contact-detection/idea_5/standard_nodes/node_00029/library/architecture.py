import torch
import torch.nn as nn
from library.config import Config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP).

    A simple but high-capacity MLP designed to process flattened temporal windows of
    kinematic data. This architecture is preferred over CNNs/TCNs for this task
    because it allows training on the full dataset (3.4M samples) without the
    computational overhead of 1D convolutions, leading to better generalization.
    (Cite solution_lesson_node_00023)
    """

    def __init__(self):
        super(KinematicMLP, self).__init__()

        self.input_dim = Config.INPUT_WIDTH

        # Increased capacity for full dataset training
        self.hidden_dim_1 = 512
        self.hidden_dim_2 = 256
        self.hidden_dim_3 = 128

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(self.input_dim, self.hidden_dim_1),
            nn.BatchNorm1d(self.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Layer 2
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.BatchNorm1d(self.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Layer 3
            nn.Linear(self.hidden_dim_2, self.hidden_dim_3),
            nn.BatchNorm1d(self.hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Output
            nn.Linear(self.hidden_dim_3, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
