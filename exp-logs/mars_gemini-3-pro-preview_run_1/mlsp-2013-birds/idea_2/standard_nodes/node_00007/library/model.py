import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    A generic model wrapper using timm backbones with a custom pooling head.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        num_classes=19,
        dropout_rate=0.0,
    ):
        super(BirdModel, self).__init__()

        # Create backbone using timm
        # global_pool='' removes the default pooling and classifier
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        in_features = self.backbone.num_features

        # Adaptive Concatenation Pooling
        # Combines Max Pooling (for salient features) and Avg Pooling (for global context)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features * 2, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)

        # Apply concat pooling
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)

        x = self.head(x)
        return x


def get_model(
    model_name=Config.MODEL_NAME,
    pretrained=Config.PRETRAINED,
    num_classes=Config.NUM_CLASSES,
    dropout_rate=Config.DROPOUT_RATE,
    freeze_backbone=False,
):
    """
    Factory function to initialize the model.
    """
    model = BirdModel(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )

    if freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False

    return model
