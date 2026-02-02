import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability with pow
        x = x.clamp(min=eps)
        # Average pooling on x^p
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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


class WhaleModel(nn.Module):
    """
    Wrapper for timm models adapted for 1-channel spectrogram input
    and GeM pooling.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleModel, self).__init__()
        self.model_name = model_name

        # Load backbone with 3 channels initially to load pretrained weights correctly
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)

        # --- Input Adaptation (3-channel -> 1-channel) ---
        # We average the weights of the first conv layer to adapt to 1-channel input
        if "resnet" in model_name:
            self._modify_first_conv_layer(self.backbone, "conv1")
        elif "efficientnet" in model_name:
            self._modify_first_conv_layer(self.backbone, "conv_stem")
        else:
            # Fallback or generic handling could be added here,
            # but we stick to the specified architectures.
            pass

        # Determine the number of input features for the classifier
        self.in_features = self.backbone.num_features

        # --- Head Replacement ---
        # Replace standard pooling and classifier with GeM + Linear
        self.gem = GeM()
        self.head = nn.Linear(self.in_features, Config.num_classes)

    def _modify_first_conv_layer(self, backbone, layer_name):
        """
        Replaces the first convolutional layer with a 1-channel version,
        initializing weights by averaging the original 3-channel weights.
        """
        old_layer = getattr(backbone, layer_name)

        # Create new conv layer with in_channels=1
        new_layer = nn.Conv2d(
            in_channels=1,
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=(old_layer.bias is not None),
        )

        # Initialize weights by averaging across the channel dimension (dim 1)
        # Shape: (out_channels, in_channels, k_h, k_w)
        with torch.no_grad():
            new_layer.weight[:] = old_layer.weight.mean(dim=1, keepdim=True)
            if old_layer.bias is not None:
                new_layer.bias[:] = old_layer.bias

        # Replace the layer in the backbone
        setattr(backbone, layer_name, new_layer)

    def forward(self, x):
        # Extract spatial features (B, C, H, W)
        x = self.backbone.forward_features(x)

        # Apply Generalized Mean Pooling -> (B, C, 1, 1)
        x = self.gem(x)

        # Flatten -> (B, C)
        x = x.flatten(1)

        # Classifier -> (B, num_classes)
        x = self.head(x)

        return x
