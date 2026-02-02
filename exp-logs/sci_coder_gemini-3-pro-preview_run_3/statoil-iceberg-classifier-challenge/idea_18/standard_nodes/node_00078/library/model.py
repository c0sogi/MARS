import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import get_device, seed_everything


class SimpleCNN(nn.Module):
    """
    SimpleCNN
    A 4-stage CNN with Global Max Pooling and Raw Angle Fusion.
    Cite solution_lesson_node_00031: Simplicity and Inductive Bias.
    Cite solution_lesson_node_00050: Early Channel Expansion.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Configuration
        # Channels: 64 -> 128 -> 128 -> 128

        # Stage 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        # Stage 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        # Stage 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # Classification Head
        # Global Max Pool results in 128 features
        # Concatenated with 1 angle feature -> 129 inputs
        # Cite solution_lesson_node_00040: Shallow dense classifier heads
        self.fc1 = nn.Linear(129, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform / Fan-In)
        Explicitly applied to ensure consistency.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Global Max Pooling
        # Cite solution_lesson_node_00007: Prefer Global Max Pooling
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Feature Fusion
        # angle shape: (B,) -> (B, 1)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)  # (B, 129)

        # Head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """
    Returns the Adam optimizer as specified in the strategy.
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_criterion():
    """
    Returns the BCEWithLogitsLoss.
    """
    return nn.BCEWithLogitsLoss()
