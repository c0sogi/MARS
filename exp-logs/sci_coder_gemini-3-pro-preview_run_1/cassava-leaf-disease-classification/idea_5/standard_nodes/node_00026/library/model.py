import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout.
    Applies multiple dropout masks to the input features and averages the predictions
    from the shared linear layer.
    """

    def __init__(
        self, in_features: int, out_features: int, num_samples: int, dropout_rate: float
    ):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            num_samples (int): Number of dropout samples to average.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, In_Features)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))

        # Average the logits
        return out / len(self.dropouts)


class CassavaModel(nn.Module):
    """
    Cassava Leaf Disease Classification Model.
    Uses a ConvNeXt-Small backbone with a Multi-Sample Dropout head.
    """

    def __init__(self, config: Config, pretrained: bool = True):
        """
        Args:
            config (Config): Configuration object containing model hyperparameters.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super().__init__()
        self.config = config

        # Initialize backbone
        # We use num_classes to initialize the standard head structure,
        # then replace the specific linear layer.
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=pretrained,
            num_classes=config.num_classes,
            drop_path_rate=config.drop_path_rate,
            drop_rate=config.dropout_rate,
        )

        # Replace the final linear layer with MultiSampleDropout
        # ConvNeXt implementation in timm uses model.head.fc for the linear projection
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "fc"):
            in_features = self.backbone.head.fc.in_features

            if config.use_multi_sample_dropout:
                self.backbone.head.fc = MultiSampleDropout(
                    in_features=in_features,
                    out_features=config.num_classes,
                    num_samples=config.multi_sample_dropout_count,
                    dropout_rate=config.head_dropout_rate,
                )
            else:
                # Standard dropout if MSD is disabled
                self.backbone.head.fc = nn.Sequential(
                    nn.Dropout(config.head_dropout_rate),
                    nn.Linear(in_features, config.num_classes),
                )
        else:
            # Fallback for other architectures (e.g. ResNet uses .fc, EfficientNet uses .classifier)
            # This ensures the model class is robust to config changes
            target_layer_name = None
            if hasattr(self.backbone, "fc"):
                target_layer_name = "fc"
            elif hasattr(self.backbone, "classifier"):
                target_layer_name = "classifier"

            if target_layer_name:
                layer = getattr(self.backbone, target_layer_name)
                # Handle case where classifier might be a Sequential or just Linear
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                elif isinstance(layer, nn.Sequential):
                    # Assuming the last layer is Linear
                    for module in reversed(layer):
                        if isinstance(module, nn.Linear):
                            in_features = module.in_features
                            break
                else:
                    # Fallback default if structure is unclear, though usually not reached for standard timm models
                    in_features = layer.in_features

                if config.use_multi_sample_dropout:
                    setattr(
                        self.backbone,
                        target_layer_name,
                        MultiSampleDropout(
                            in_features=in_features,
                            out_features=config.num_classes,
                            num_samples=config.multi_sample_dropout_count,
                            dropout_rate=config.head_dropout_rate,
                        ),
                    )
                else:
                    setattr(
                        self.backbone,
                        target_layer_name,
                        nn.Linear(in_features, config.num_classes),
                    )

    def forward(self, x):
        return self.backbone(x)
