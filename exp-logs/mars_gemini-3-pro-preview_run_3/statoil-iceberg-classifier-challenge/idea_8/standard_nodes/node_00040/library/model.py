import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    A simple 4-layer CNN with Global Max Pooling and Late Fusion of incidence angle.
    Optimized for small, noisy radar datasets.
    Cite Lesson 00039: Late Fusion is preferred over complex modulation for small datasets.
    Cite Lesson 00026: Capped channel width (128) prevents overfitting compared to VGG scaling.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input: 3 channels (HH, HV, Avg)
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Dense Layers
        # Global Max Pooling results in 128 features.
        # Concatenating incidence angle adds 1 feature -> 129.
        self.fc1 = nn.Linear(129, Config.FC_DIM)
        self.fc2 = nn.Linear(Config.FC_DIM, 256)
        self.fc3 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Feature Extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Global Max Pooling (Cite Lesson 00007: Max Pooling captures peak signal better than Avg)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)  # (B, 128)

        # Late Fusion (Cite Lesson 00039)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)  # (B, 129)

        # Classification Head
        x = F.relu(self.fc1(x))
        # Cite Lesson 00017: Apply dropout after first dense activation, not before
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x
