import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    This layer computes the generalized mean of the spatial dimensions of the input tensor.
    It is defined as: f(X) = (1/|X| * sum(x^p))^(1/p).

    When p=1, this is equivalent to Average Pooling.
    When p -> infinity, this approaches Max Pooling.
    The parameter 'p' is learnable, allowing the model to adaptively select the pooling strategy.

    Attributes:
        p (torch.nn.Parameter): The learnable power parameter (initialized to 3.0).
        eps (float): A small constant for numerical stability to avoid NaN gradients.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the GeM layer.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Pooled output of shape (Batch, Channels, 1, 1).
        """
        # Clamp input to epsilon to ensure non-negative base for power operation
        # This handles cases where activation functions (like GELU) might produce negative values
        x = x.clamp(min=self.eps)

        # Apply the GeM formula: (AvgPool(x^p))^(1/p)
        # adaptive_avg_pool2d ensures the spatial dimensions are reduced to 1x1 regardless of input size
        x_p = x.pow(self.p)
        x_p_avg = F.adaptive_avg_pool2d(x_p, output_size=(1, 1))

        return x_p_avg.pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout (MSD) layer.

    This layer implements a technique to regularize the model and minimize log loss.
    Instead of a single dropout mask followed by a linear layer, it applies multiple
    dropout masks (samples) to the input features. Each masked input is passed through
    the *same* shared linear layer, and the resulting logits are averaged.

    This acts as an internal ensemble during training and inference, providing smoother
    probability estimates and faster convergence.

    Attributes:
        dropouts (nn.ModuleList): A list of independent Dropout layers.
        fc (nn.Linear): The shared fully connected layer for classification.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        """
        Args:
            in_features (int): Size of the input feature vector.
            out_features (int): Size of the output vector (number of classes or 1 for binary).
            num_samples (int): Number of dropout masks to apply.
            dropout_rate (float): Probability of an element to be zeroed.
        """
        super(MultiSampleDropout, self).__init__()

        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Forward pass of the Multi-Sample Dropout layer.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, out_features).
        """
        logits_list = []

        # Apply each dropout mask and pass through the shared linear layer
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(x)))

        # Stack the results [num_samples, Batch, out_features] and average across the sample dimension
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)
