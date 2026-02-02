import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with Global Max Pooling and Late Fusion.
    Optimized for small, noisy datasets by using aggressive spatial reduction.
    Cite solution_lesson_node_00046
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Configuration
        in_channels = Config.IN_CHANNELS
        base_filters = Config.BASE_FILTERS
        max_filters = Config.MAX_FILTERS
        dropout_fc = Config.DROPOUT_FC

        # ---------------------------------------------------------------------
        # Convolutional Backbone
        # ---------------------------------------------------------------------
        # Block 1: 75x75 -> 37x37
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 37x37 -> 18x18
        self.block2 = nn.Sequential(
            nn.Conv2d(base_filters, max_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(max_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 18x18 -> 9x9
        self.block3 = nn.Sequential(
            nn.Conv2d(max_filters, max_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(max_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 9x9 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(max_filters, max_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(max_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        # Global Max Pooling implies output is 1x1 x Channels
        # Feature vector size = max_filters (128)

        # Late Fusion: 128 features + 1 angle
        self.fc_input_dim = max_filters + 1
        self.hidden_dim = 512  # Cite solution_lesson_node_00040

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),  # Cite solution_lesson_node_00017
            nn.Linear(self.hidden_dim, 1),
        )

        # Initialize weights (using PyTorch defaults implicitly, or explicit Kaiming)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle tensor (B, 1) or (B,)
        """
        # Ensure angle has correct shape (B, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Feature Extraction
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # Global Max Pooling (Cite solution_lesson_node_00005)
        # Returns (B, C, 1, 1) -> Flatten to (B, C)
        out = F.adaptive_max_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Late Fusion (Cite solution_lesson_node_00039)
        out = torch.cat([out, angle], dim=1)

        # Classification
        logits = self.classifier(out)

        return logits
