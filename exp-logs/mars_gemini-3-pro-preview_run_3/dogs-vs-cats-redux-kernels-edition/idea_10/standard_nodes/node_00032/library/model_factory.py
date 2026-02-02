import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger("model_factory")


class MultiSampleDropoutHead(nn.Module):
    """
    A custom classification head that applies multiple dropout masks to the input features
    and averages the predictions. This acts as an internal ensemble, smoothing the
    loss landscape and reducing log loss.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        """
        Args:
            in_features (int): Number of input features from the backbone.
            out_features (int): Number of output classes (1 for binary).
            dropout_rates (list of float): List of dropout probabilities to apply.
        """
        super().__init__()
        self.dropout_rates = dropout_rates
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        logits = []
        for rate in self.dropout_rates:
            # Apply dropout with the specific rate
            # F.dropout handles the training=self.training logic automatically
            # (i.e., it is identity during eval)
            out = F.dropout(x, p=rate, training=self.training)
            out = self.fc(out)
            logits.append(out)

        # Stack the results from all dropout masks and calculate the mean
        # Shape: [batch_size, out_features]
        return torch.stack(logits).mean(dim=0)


def create_model(model_name, pretrained=True):
    """
    Creates a model instance based on the configuration.

    Args:
        model_name (str): The name of the model to create (must be in Config.MODEL_SPECS).
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        nn.Module: The constructed PyTorch model.
    """
    if model_name not in Config.MODEL_SPECS:
        raise ValueError(
            f"Model {model_name} not found in config. Available: {list(Config.MODEL_SPECS.keys())}"
        )

    spec = Config.MODEL_SPECS[model_name]
    timm_name = spec["timm_name"]

    logger.info(
        f"Creating model: {model_name} (timm: {timm_name}) | Pretrained: {pretrained}"
    )

    # Create the backbone using timm
    # num_classes=1 for binary classification (BCEWithLogitsLoss)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)

    # Apply Multi-Sample Dropout Head if enabled in Config
    if Config.USE_MULTI_SAMPLE_DROPOUT:
        _replace_head_with_msd(model, model_name)

    return model


def _replace_head_with_msd(model, model_name):
    """
    Helper function to locate the final linear layer and replace it with
    MultiSampleDropoutHead.

    Args:
        model (nn.Module): The model instance.
        model_name (str): The model key to handle architecture-specific head paths.
    """
    dropout_rates = Config.DROPOUT_RATES
    logger.info(
        f"Replacing classifier with MultiSampleDropoutHead (Rates: {dropout_rates})"
    )

    # Architecture-specific logic to find the linear layer
    # ResNet-style: model.fc
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = MultiSampleDropoutHead(in_features, 1, dropout_rates)
        return

    # ConvNeXt / MaxViT / Swin style: model.head.fc
    # Many modern timm models encapsulate the head in a class (e.g., NormMlpClassifierHead)
    # where the linear layer is 'fc' inside 'head'.
    if hasattr(model, "head"):
        if hasattr(model.head, "fc") and isinstance(model.head.fc, nn.Linear):
            in_features = model.head.fc.in_features
            model.head.fc = MultiSampleDropoutHead(in_features, 1, dropout_rates)
            return
        # Some architectures might have model.head as the Linear layer itself
        elif isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = MultiSampleDropoutHead(in_features, 1, dropout_rates)
            return

    # Fallback / Error
    logger.warning(
        f"Could not automatically locate linear layer for {model_name}. "
        "Multi-Sample Dropout was NOT applied."
    )
