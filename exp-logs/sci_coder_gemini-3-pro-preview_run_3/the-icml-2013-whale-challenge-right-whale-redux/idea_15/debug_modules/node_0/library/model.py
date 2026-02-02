import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean over the spatial dimensions.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (Batch, Channels, Height, Width)
        # Apply clamping to avoid NaN in power operation
        # Average pooling over spatial dimensions (H, W)
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


class WhaleEfficientNet(nn.Module):
    """
    EfficientNetV2-Medium backbone with GeM pooling and dynamic regularization.

    This model supports the 'Calibrated Noisy Student' strategy by allowing
    independent configuration of:
    - drop_rate: Dropout rate for the final classifier layer.
    - drop_path_rate: Stochastic Depth rate for the backbone blocks.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        drop_rate=0.0,  # Classifier Dropout Rate
        drop_path_rate=0.0,  # Stochastic Depth Rate (Backbone)
    ):
        super(WhaleEfficientNet, self).__init__()

        # Initialize backbone using timm
        # num_classes=0 and global_pool='' allows us to access the feature maps directly
        # drop_path_rate controls Stochastic Depth in the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )

        # Dynamically determine the number of input features
        # This ensures compatibility if the backbone model changes
        with torch.no_grad():
            # Create a dummy input matching the expected input shape (except batch size)
            # EfficientNet usually expects 224x224 or similar, but is flexible
            dummy_input = torch.randn(1, in_channels, 224, 224)
            features = self.backbone(dummy_input)
            self.in_features = features.shape[1]

        # Generalized Mean Pooling Layer
        self.pooling = GeM()

        # Classification Head
        # Dropout is applied before the final linear layer
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        # Feature Extraction
        # Shape: (Batch, Channels, Height, Width)
        x = self.backbone(x)

        # Pooling
        # Shape: (Batch, Channels, 1, 1)
        x = self.pooling(x)

        # Flatten
        # Shape: (Batch, Channels)
        x = x.flatten(1)

        # Regularization (Dropout)
        x = self.drop(x)

        # Classification
        # Shape: (Batch, Num_Classes)
        x = self.fc(x)

        return x
