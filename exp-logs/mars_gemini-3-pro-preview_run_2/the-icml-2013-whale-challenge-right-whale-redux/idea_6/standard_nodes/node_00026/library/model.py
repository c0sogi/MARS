import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (Avg(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp for numerical stability
        x = x.clamp(min=eps)
        # Apply average pooling to x^p
        # Kernel size is the spatial size of the input (H, W)
        avg_pool = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1)))
        # Raise to power 1/p
        return avg_pool.pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class WhaleEnsembleMember(nn.Module):
    """
    A single member of the ensemble (EfficientNet or ResNet).
    Adapts the first layer for 1-channel input and uses GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleEnsembleMember, self).__init__()

        # Load backbone with no classifier and no global pooling (returns feature maps)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Modify first convolutional layer to accept 1 channel
        self._modify_first_layer(model_name)

        # Generalized Mean Pooling
        self.pooling = GeM()

        # Classifier Head
        # timm models expose num_features attribute
        self.head = nn.Linear(self.backbone.num_features, Config.NUM_CLASSES)

    def _modify_first_layer(self, model_name):
        # Identify the first layer based on architecture conventions
        first_layer_name = None
        if "efficientnet" in model_name:
            first_layer_name = "conv_stem"
        elif "resnet" in model_name:
            first_layer_name = "conv1"

        # Fallback: find first Conv2d if name convention fails
        if first_layer_name is None or not hasattr(self.backbone, first_layer_name):
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    first_layer_name = name
                    break

        if first_layer_name:
            old_layer = getattr(self.backbone, first_layer_name)

            # Create new layer with in_channels=1
            # Preserve all other attributes (stride, padding, etc.)
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

            # Initialize weights by averaging the original RGB weights
            # Shape: (Out, In, K, K)
            with torch.no_grad():
                new_layer.weight[:] = torch.mean(old_layer.weight, dim=1, keepdim=True)
                if old_layer.bias is not None:
                    new_layer.bias[:] = old_layer.bias

            # Replace the layer in the backbone
            setattr(self.backbone, first_layer_name, new_layer)
        else:
            raise AttributeError(
                f"Could not identify first convolutional layer for {model_name}"
            )

    def forward(self, x):
        # Input x: (Batch, 1, F, T)

        # Feature extraction
        features = self.backbone(x)  # (Batch, C, H, W)

        # Pooling
        pooled = self.pooling(features)  # (Batch, C, 1, 1)
        pooled = pooled.flatten(1)  # (Batch, C)

        # Classification
        logits = self.head(pooled)  # (Batch, 1)

        return logits
