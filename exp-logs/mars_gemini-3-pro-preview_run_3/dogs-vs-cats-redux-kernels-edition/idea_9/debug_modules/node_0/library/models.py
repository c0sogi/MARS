import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x is expected to be (B, C, H, W)
        # Clamp for numerical stability
        x = x.clamp(min=eps).pow(p)

        # Average pooling over spatial dimensions (H, W)
        # output becomes (B, C, 1, 1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Power 1/p
        return x.pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class PetModel(nn.Module):
    """
    Wrapper class to implement the Triple Heterogeneous Ensemble architectures.
    Handles the specific pooling and head modifications for ConvNeXt, ResNet, and Swin.
    """

    def __init__(self, model_name: str, num_classes: int = 1, pretrained: bool = True):
        super(PetModel, self).__init__()
        self.model_name = model_name

        # Map simple names to specific timm model keys
        # Using 'fb_in1k' for ConvNeXt and 'a1_in1k' for ResNet as modern robust weights
        # Using 'ms_in1k' for Swin
        if model_name == "convnext_base":
            timm_name = "convnext_base.fb_in1k"
            self.use_custom_head = True
        elif model_name == "resnet101":
            timm_name = "resnet101.a1_in1k"
            self.use_custom_head = True
        elif model_name == "swin_s":
            timm_name = "swin_small_patch4_window7_224.ms_in1k"
            self.use_custom_head = False
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Choose from ['convnext_base', 'resnet101', 'swin_s']"
            )

        if self.use_custom_head:
            # For ConvNeXt and ResNet, we remove the default pooling and head
            # to insert GeM pooling.
            # num_classes=0 and global_pool='' returns the feature map (B, C, H, W)
            self.backbone = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=0, global_pool=""
            )

            # Determine the number of input features for the head
            num_features = self.backbone.num_features

            # Custom Head: GeM -> Flatten -> Linear
            self.pool = GeM(p=3)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(num_features, num_classes)

        else:
            # For Swin Transformer, we retain the native attention-based pooling mechanism.
            # We simply initialize it with the correct number of classes.
            self.backbone = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=num_classes
            )
            # Identity layers for consistency in forward pass structure if needed,
            # though we will branch in forward().
            self.pool = nn.Identity()
            self.flatten = nn.Identity()
            self.fc = nn.Identity()

    def forward(self, x):
        if self.use_custom_head:
            # Extract features: (B, C, H, W)
            x = self.backbone(x)

            # Apply GeM Pooling: (B, C, 1, 1)
            x = self.pool(x)

            # Flatten: (B, C)
            x = self.flatten(x)

            # Classification
            x = self.fc(x)
        else:
            # Swin Transformer handles pooling/classification internally
            x = self.backbone(x)

        return x


def get_model(model_name: str, num_classes: int = 1, pretrained: bool = True):
    """
    Factory function to create the model.

    Args:
        model_name (str): 'convnext_base', 'resnet101', or 'swin_s'
        num_classes (int): Number of output classes (default 1 for binary)
        pretrained (bool): Whether to load pretrained ImageNet weights

    Returns:
        nn.Module: The configured PyTorch model.
    """
    return PetModel(
        model_name=model_name, num_classes=num_classes, pretrained=pretrained
    )
