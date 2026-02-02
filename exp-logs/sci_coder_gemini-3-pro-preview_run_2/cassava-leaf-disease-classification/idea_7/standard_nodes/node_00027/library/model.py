import torch.nn as nn
import timm
from library.config import Config


def get_model(config: Config) -> nn.Module:
    """
    Instantiates the ConvNeXt model based on the provided configuration.

    Uses timm.create_model to load the architecture and pre-trained weights.
    Configures the classification head and regularization parameters.

    Args:
        config (Config): Configuration object containing model parameters
                         (model_name, num_classes, drop_path_rate, dropout_rate).

    Returns:
        nn.Module: The PyTorch model ready for training.
    """
    # Create the model using timm
    # pretrained=True downloads/loads weights from the hub (cached locally if available)
    # num_classes replaces the head with a new linear layer for our specific task
    # drop_path_rate sets the stochastic depth regularization
    # drop_rate sets the dropout rate for the classifier head
    model = timm.create_model(
        model_name=config.model_name,
        pretrained=True,
        num_classes=config.num_classes,
        drop_path_rate=config.drop_path_rate,
        drop_rate=config.dropout_rate,
    )

    return model
