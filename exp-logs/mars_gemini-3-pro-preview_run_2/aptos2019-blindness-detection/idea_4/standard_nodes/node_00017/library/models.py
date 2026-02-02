import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the input feature map.
    Formula: f(X) = (1/N * sum(X^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter, initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp input to eps to avoid numerical instability with pow
        x = torch.clamp(x, min=self.eps)

        # Calculate average of x^p
        # F.avg_pool2d with kernel_size=(H, W) performs global averaging
        x_pow = x.pow(self.p)
        avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Take the p-th root
        return avg.pow(1.0 / self.p)


class RetinopathyModel(nn.Module):
    """
    Retinopathy Severity Prediction Model.
    Wraps a timm backbone with GeM pooling and a linear regression head.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the architecture (e.g., 'tf_efficientnet_b5_ns').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(RetinopathyModel, self).__init__()

        # Load the backbone from timm
        # num_classes=0 and global_pool='' ensures we get the spatial feature map
        # instead of the classification logits or a pooled vector.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input channels for the head
        # Most timm models have a num_features attribute
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: perform a dummy forward pass to infer shape
            with torch.no_grad():
                # Use a small dummy input
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Pooling layer
        self.pooling = GeM(p=3)

        # Regression Head
        # Projects features to a single continuous scalar (severity score)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W)

        Returns:
            torch.Tensor: Predicted severity scores of shape (Batch,)
        """
        # 1. Extract features
        # Shape: (Batch, Channels, H_feat, W_feat)
        features = self.backbone(x)

        # 2. Apply GeM Pooling
        # Shape: (Batch, Channels, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Shape: (Batch, Channels)
        flattened = pooled.view(pooled.size(0), -1)

        # 4. Regression Prediction
        # Shape: (Batch, 1)
        output = self.fc(flattened)

        # Flatten to (Batch,) to match the target shape in the training loop
        return output.view(-1)
