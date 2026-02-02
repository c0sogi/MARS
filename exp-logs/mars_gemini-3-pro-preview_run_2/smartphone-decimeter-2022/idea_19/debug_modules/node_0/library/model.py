import torch
import torch.nn as nn
from library.config import config


class ECM_MLP(nn.Module):
    def __init__(self):
        super(ECM_MLP, self).__init__()

        # Calculate input dimension dynamically based on config
        # Trajectory block: Window size * number of trajectory features per epoch
        self.traj_dim = config.WINDOW_SIZE * len(config.TRAJ_FEATURES)

        # Context blocks: Aggregated features (1 set per window)
        self.env_dim = len(config.ENV_FEATURES)
        self.imu_dim = len(config.IMU_FEATURES)

        self.input_dim = self.traj_dim + self.env_dim + self.imu_dim

        # Model Architecture
        layers = []

        # Input Layer
        layers.append(nn.Linear(self.input_dim, config.HIDDEN_DIM))
        layers.append(nn.BatchNorm1d(config.HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.DROPOUT))

        # Hidden Layers
        # We already added 1 layer, and the last is output, so we add num_layers - 2 hidden blocks
        for _ in range(config.NUM_LAYERS - 2):
            layers.append(nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM))
            layers.append(nn.BatchNorm1d(config.HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT))

        # Output Layer (2D: North residual, East residual)
        layers.append(nn.Linear(config.HIDDEN_DIM, 2))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               The input is expected to be the flattened concatenation of:
               [Trajectory Features (Windowed), Env Context, IMU Context]
        Returns:
            out: Output tensor of shape (batch_size, 2) representing (delta_north, delta_east)
        """
        return self.net(x)
