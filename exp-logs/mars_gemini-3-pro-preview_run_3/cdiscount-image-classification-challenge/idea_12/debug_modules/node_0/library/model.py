import torch
import torch.nn as nn
import library.config as config


class HierarchicalMLP(nn.Module):
    """
    Multi-Task MLP for hierarchical product categorization.

    Architecture:
        - Input: Concatenated features from DualBackbone (ResNet50 + EfficientNetB0).
        - Trunk: Shared fully connected layers with BatchNorm and Dropout to extract
                 high-level representations from the fused embeddings.
        - Heads: Three separate linear heads for Level 1, Level 2, and Level 3 categorization.
    """

    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE,
        num_classes_l1=config.NUM_CLASSES_L1,
        num_classes_l2=config.NUM_CLASSES_L2,
        num_classes_l3=config.NUM_CLASSES_L3,
    ):
        """
        Args:
            input_dim (int): Dimensionality of input features (default: 3328).
            hidden_dim (int): Dimensionality of hidden layers (default: 1024).
            dropout_rate (float): Dropout probability (default: 0.3).
            num_classes_l1 (int): Number of Level 1 categories.
            num_classes_l2 (int): Number of Level 2 categories.
            num_classes_l3 (int): Number of Level 3 categories (Target).
        """
        super(HierarchicalMLP, self).__init__()

        # Shared Feature Extraction Trunk
        # Processes the raw feature vector into a shared latent representation
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

        # Independent Classification Heads
        # Each head projects the shared representation to the specific class logits
        self.head_l1 = nn.Linear(hidden_dim, num_classes_l1)
        self.head_l2 = nn.Linear(hidden_dim, num_classes_l2)
        self.head_l3 = nn.Linear(hidden_dim, num_classes_l3)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal for ReLUs and Constant for BatchNorm.
        """
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
        Forward pass of the hierarchical model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Input_Dim).

        Returns:
            tuple: (logits_l1, logits_l2, logits_l3)
        """
        # Pass through shared trunk
        features = self.trunk(x)

        # Pass through independent heads
        logits_l1 = self.head_l1(features)
        logits_l2 = self.head_l2(features)
        logits_l3 = self.head_l3(features)

        return logits_l1, logits_l2, logits_l3
