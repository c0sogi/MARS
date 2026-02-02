import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean of the spatial features:
    f(X) = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter p. Default: 3.0.
        eps (float): Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, 1, 1).
        """
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp min value to avoid NaN gradients with pow
        x = x.clamp(min=eps)
        # Apply GeM formula: (AvgPool(x^p))^(1/p)
        # We use avg_pool2d over the spatial dimensions (H, W)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class CosineClassifier(nn.Module):
    """
    Cosine Similarity Classifier Head.

    Performs classification based on cosine similarity between feature vectors and class weight vectors.
    Both inputs and weights are normalized to the unit hypersphere.

    logits = scale * (x_norm . W_norm)

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Number of classes.
        scale (float): Initial value for the scaling factor (temperature). Default: 30.0.
    """

    def __init__(self, in_features, out_features, scale=30.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix [out_features, in_features]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.Tensor([scale]))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming Uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input features of shape (B, in_features).

        Returns:
            torch.Tensor: Logits of shape (B, out_features).
        """
        # Normalize input features (L2 norm)
        x_norm = F.normalize(input, p=2, dim=1)

        # Normalize weight vectors (L2 norm)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Calculate Cosine Similarity
        # F.linear computes xW^T. With normalized inputs, this is the cosine similarity.
        cosine_sim = F.linear(x_norm, w_norm)

        # Apply scaling factor
        logits = self.scale * cosine_sim

        return logits

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(in_features={self.in_features}, out_features={self.out_features}, scale={self.scale.item():.2f})"
        )
