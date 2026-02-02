import torch
import torch.nn as nn
from library.config import Config


class DeepFeatureCascade(nn.Module):
    """
    Deep Feature Cascading (DFC) Network.

    This architecture explicitly models the hierarchical structure of the product categories.
    It processes features in three stages (L1 -> L2 -> L3), where each stage receives
    both the original visual features and the semantic hidden state from the previous stage.
    """

    def __init__(self):
        super(DeepFeatureCascade, self).__init__()

        # ==========================
        # Configuration
        # ==========================
        self.input_dim = Config.INPUT_DIM  # 3328
        self.hidden_dim = Config.HIDDEN_DIM  # 1024
        self.dropout_rate = Config.DROPOUT_RATE  # 0.3

        self.num_l1 = Config.NUM_CLASSES_L1  # 49
        self.num_l2 = Config.NUM_CLASSES_L2  # 483
        self.num_l3 = Config.NUM_CLASSES_L3  # 5270

        # ==========================
        # Block 1: Level 1 (Coarse)
        # ==========================
        # Input: Raw Features
        # Output: L1 Logits + Hidden State 1
        self.b1_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.b1_head = nn.Linear(self.hidden_dim, self.num_l1)

        # ==========================
        # Block 2: Level 2 (Intermediate)
        # ==========================
        # Input: Raw Features + Hidden State 1
        # Output: L2 Logits + Hidden State 2
        self.b2_input_dim = self.input_dim + self.hidden_dim
        self.b2_layer = nn.Sequential(
            nn.Linear(self.b2_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.b2_head = nn.Linear(self.hidden_dim, self.num_l2)

        # ==========================
        # Block 3: Level 3 (Fine - Target)
        # ==========================
        # Input: Raw Features + Hidden State 2
        # Output: L3 Logits
        self.b3_input_dim = self.input_dim + self.hidden_dim
        self.b3_layer = nn.Sequential(
            nn.Linear(self.b3_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.b3_head = nn.Linear(self.hidden_dim, self.num_l3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for ReLU networks to maintain variance.
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
        Forward pass through the cascade.

        Args:
            x (torch.Tensor): Input features of shape (Batch_Size, Input_Dim)

        Returns:
            tuple: (l1_logits, l2_logits, l3_logits)
        """
        # ----------------------
        # Block 1
        # ----------------------
        h1 = self.b1_layer(x)
        l1_logits = self.b1_head(h1)

        # ----------------------
        # Block 2
        # ----------------------
        # Skip Connection: Concatenate original input with L1 hidden state
        # This allows L2 to see both the raw pixels (texture/shape) and the L1 concept
        in2 = torch.cat([x, h1], dim=1)
        h2 = self.b2_layer(in2)
        l2_logits = self.b2_head(h2)

        # ----------------------
        # Block 3
        # ----------------------
        # Skip Connection: Concatenate original input with L2 hidden state
        in3 = torch.cat([x, h2], dim=1)
        h3 = self.b3_layer(in3)
        l3_logits = self.b3_head(h3)

        return l1_logits, l2_logits, l3_logits
