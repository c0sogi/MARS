import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # (B, C, H, W) -> (B, C, 1, 1)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class WhaleModel(nn.Module):
    """
    Heterogeneous Whale Detection Model.
    Wraps a timm backbone with GeM pooling and a custom classifier head.
    Adapts 3-channel pretrained weights to 1-channel input via averaging.
    """

    def __init__(self, model_name, pretrained=True):
        super(WhaleModel, self).__init__()
        self.model_name = model_name

        # Initialize backbone
        # in_chans=1 tells timm to adapt the first layer.
        # By default, timm sums the weights (R+G+B).
        # We will correct this to average (R+G+B)/3 in _fix_first_conv_weights.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            global_pool="",  # Return feature maps, not pooled vector
            num_classes=0,  # Remove classification head
        )

        # Apply weight averaging correction
        self._fix_first_conv_weights()

        # Determine input features for the head dynamically
        in_features = self._get_in_features()

        # Classification Head
        self.pooling = GeM()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def _fix_first_conv_weights(self):
        """
        Adjusts the first convolutional layer's weights.
        timm sums weights for in_chans=1 conversion. We divide by 3 to achieve averaging.
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                with torch.no_grad():
                    module.weight.div_(3.0)
                # Break after modifying the first conv layer (entry point)
                break

    def _get_in_features(self):
        """
        Performs a dummy forward pass to calculate the output feature dimension.
        """
        with torch.no_grad():
            # Create dummy input: (Batch, Channel, Freq, Time)
            # Time dimension is approximate (2.0s * 2000Hz / 64 hop ~ 63 frames)
            dummy_input = torch.randn(2, 1, Config.N_MELS, 64)
            features = self.backbone(dummy_input)
            return features.shape[1]

    def forward(self, x):
        # Feature Extraction
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # Generalized Mean Pooling
        # Shape: (B, C, 1, 1)
        pooled = self.pooling(features)

        # Flatten
        # Shape: (B, C)
        flattened = pooled.flatten(1)

        # Dropout & Classification
        dropped = self.drop(flattened)
        output = self.fc(dropped)

        return output
