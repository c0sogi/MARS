import torch
import torch.nn as nn
from library.config import Config


class HierarchicalMLP(nn.Module):
    """
    A Multi-Task Multi-Layer Perceptron (MLP) designed for hierarchical product categorization.

    This model takes a fixed-size embedding (e.g., from a ResNet50 backbone) and passes it
    through a shared sequence of fully connected layers (trunk). The output of the trunk
    is then fed into three independent classification heads, each predicting a different
    level of the product category hierarchy (Level 1, Level 2, and Level 3).

    This architecture enforces the learning of representations that are consistent across
    different levels of granularity, acting as a structural regularizer.
    """

    def __init__(
        self,
        input_dim=Config.EMBEDDING_DIM,
        hidden_dims=[1024, 512],
        dropout_rate=0.5,
        num_classes_l1=Config.NUM_CLASSES_L1,
        num_classes_l2=Config.NUM_CLASSES_L2,
        num_classes_l3=Config.NUM_CLASSES_L3,
    ):
        """
        Args:
            input_dim (int): Dimension of the input feature vector (default: 2048).
            hidden_dims (list of int): Dimensions of the hidden layers in the shared trunk.
            dropout_rate (float): Probability of an element to be zeroed in Dropout layers.
            num_classes_l1 (int): Number of classes in Level 1 (Coarse).
            num_classes_l2 (int): Number of classes in Level 2 (Sub-category).
            num_classes_l3 (int): Number of classes in Level 3 (Fine-grained/Target).
        """
        super(HierarchicalMLP, self).__init__()

        # ---------------------------------------------------------
        # Shared Trunk Construction
        # ---------------------------------------------------------
        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim

        self.trunk = nn.Sequential(*layers)

        # ---------------------------------------------------------
        # Hierarchical Classification Heads
        # ---------------------------------------------------------
        # Head for Level 1 (e.g., "SPORT", "AUTO - MOTO")
        self.head_l1 = nn.Linear(current_dim, num_classes_l1)

        # Head for Level 2 (e.g., "CYCLES", "PIECES")
        self.head_l2 = nn.Linear(current_dim, num_classes_l2)

        # Head for Level 3 (Target, e.g., "VELO DE VILLE")
        self.head_l3 = nn.Linear(current_dim, num_classes_l3)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: A tuple containing logits for each hierarchy level:
                   (logits_l1, logits_l2, logits_l3)
        """
        # Pass input through the shared trunk
        # Shape: (batch_size, last_hidden_dim)
        features = self.trunk(x)

        # Compute logits for each level using the shared features
        logits_l1 = self.head_l1(features)
        logits_l2 = self.head_l2(features)
        logits_l3 = self.head_l3(features)

        return logits_l1, logits_l2, logits_l3
