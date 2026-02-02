import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    This layer computes the generalized mean of the spatial dimensions of the input tensor.
    It introduces a learnable parameter 'p' which interpolates between Average Pooling (p=1)
    and Max Pooling (p -> infinity).

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    This is particularly effective for sparse signals like bird calls in spectrograms,
    allowing the model to focus on high-activation regions without discarding context
    as aggressively as Max Pooling.
    """

    def __init__(self, p=3.0, eps=1e-6):
        """
        Args:
            p (float): Initial value for the power parameter. Default: 3.0.
            eps (float): Small constant for numerical stability. Default: 1e-6.
        """
        super(GeM, self).__init__()
        # p is a learnable parameter
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
        # Clamp input to eps to ensure stability before power operation
        # Note: We use the spatial dimensions of x (H, W) for the pooling kernel
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

    def __repr__(self):
        return self.__class__.__name__ + "(p={:.4f}, eps={})".format(
            self.p.data.tolist()[0], self.eps
        )
