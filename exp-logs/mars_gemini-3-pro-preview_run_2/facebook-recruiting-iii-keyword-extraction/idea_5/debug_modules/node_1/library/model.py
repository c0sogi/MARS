import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for dense multi-label classification.

    Addresses class imbalance by down-weighting easy examples (background).
    Formula: L = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weighting factor for the positive class (0 < alpha < 1).
        gamma (float): Focusing parameter to down-weight easy examples.
        reduction (str): 'mean' or 'sum' reduction for the output loss.
    """

    def __init__(self, alpha=None, gamma=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else Config.FOCAL_ALPHA
        self.gamma = gamma if gamma is not None else Config.FOCAL_GAMMA
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape (N, C).
            targets (torch.Tensor): Multi-hot binary targets of shape (N, C).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Numerical stability: use binary_cross_entropy_with_logits to compute standard BCE
        # This computes -y*log(p) - (1-y)*log(1-p)
        # We need the individual terms to apply the focal weights.

        # Calculate probabilities from logits
        probs = torch.sigmoid(inputs)

        # Calculate the focal weights (modulating factor)
        # pt is the probability of the true class
        # If y=1, pt = p; if y=0, pt = 1-p
        # weight = (1 - pt)^gamma

        # For targets == 1: weight = (1 - p)^gamma
        # For targets == 0: weight = p^gamma
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Apply weighting factor
        focal_weight = (1 - pt).pow(self.gamma)

        # Apply alpha balancing
        # For targets == 1: alpha
        # For targets == 0: 1 - alpha
        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        # Compute standard BCE loss (element-wise)
        # reduction='none' is crucial here so we can multiply by weights first
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Combine
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LinearTaggingModel(nn.Module):
    """
    GPU-Accelerated Sparse Linear Network.

    A simple but effective architecture for high-dimensional sparse data.
    Consists of a single linear layer mapping input features to output tags.

    Args:
        input_dim (int): Dimension of the input feature vector (vocab size).
        output_dim (int): Dimension of the output vector (number of tags).
    """

    def __init__(self, input_dim=None, output_dim=None):
        super(LinearTaggingModel, self).__init__()

        self.input_dim = input_dim if input_dim is not None else Config.INPUT_DIM
        self.output_dim = output_dim if output_dim is not None else Config.OUTPUT_DIM

        # Single fully connected layer
        # Bias is True by default, which is appropriate here
        self.fc = nn.Linear(self.input_dim, self.output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        Good for linear layers with sigmoid activation later.
        """
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim).
        """
        return self.fc(x)
