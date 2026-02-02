import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to eps to avoid NaN gradients with non-integer p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

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


class WhaleClassifier(nn.Module):
    """
    Classifier for Right Whale Detection.
    Wraps a timm backbone, adapts the first layer for 1-channel input,
    and applies GeM pooling followed by a linear head.
    """

    def __init__(self, model_name, pretrained=True, in_channels=1, num_classes=1):
        super(WhaleClassifier, self).__init__()

        # Load backbone with no classifier and no global pooling to get raw feature maps
        # We load with in_chans=3 initially to ensure we get the correct pretrained weights structure
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )

        # Adapt the first layer if input channels differ (e.g., 1 channel spectrogram)
        if in_channels != 3:
            self._adapt_first_layer(in_channels, model_name)

        # Pooling and Classification Head
        self.gem = GeM()
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def _adapt_first_layer(self, in_channels, model_name):
        """
        Replaces the first convolutional layer to accept `in_channels`.
        Weights are initialized by averaging the pretrained weights across the channel dimension.
        """
        # Identify the first layer based on architecture
        if "resnet" in model_name:
            first_conv = self.backbone.conv1
            parent_module = self.backbone
            layer_name = "conv1"
        elif "efficientnet" in model_name:
            first_conv = self.backbone.conv_stem
            parent_module = self.backbone
            layer_name = "conv_stem"
        else:
            # Fallback: try to find the first Conv2d module
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    parent_module = self.backbone
                    layer_name = name.split(".")[0]  # Approximation for top-level
                    break

        # Create a new Conv2d layer with the target in_channels
        # We attempt to preserve kernel, stride, padding, and bias configuration
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None),
        )

        # Initialize weights by averaging the original weights
        with torch.no_grad():
            # Original weight shape: (Out, In_Original, K, K)
            # Average across the input channel dimension (dim=1)
            new_weight = torch.mean(first_conv.weight, dim=1, keepdim=True)
            new_conv.weight.copy_(new_weight)

            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Replace the layer in the backbone
        setattr(parent_module, layer_name, new_conv)

    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)

        # GeM Pooling
        x = self.gem(x)

        # Flatten: (Batch, Channels, 1, 1) -> (Batch, Channels)
        x = torch.flatten(x, 1)

        # Classification
        x = self.head(x)

        return x
