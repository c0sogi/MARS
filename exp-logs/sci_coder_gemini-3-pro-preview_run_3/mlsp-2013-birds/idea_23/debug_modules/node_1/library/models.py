import torch
import torch.nn as nn
import torchvision.models as models


def get_model(model_name, config, device="cpu"):
    """
    Factory function to initialize and return a PyTorch model with a modified head
    for the specific number of classes defined in the config.

    Args:
        model_name (str): Name of the architecture ('resnet18', 'efficientnet_b0', 'densenet121').
        config (Config): Configuration object containing NUM_CLASSES.
        device (str or torch.device): Device to move the model to.

    Returns:
        torch.nn.Module: The initialized model moved to the specified device.
    """
    num_classes = config.NUM_CLASSES

    # Normalize model name
    model_name = model_name.lower()

    if model_name == "resnet18":
        # Load ResNet18 with default pre-trained weights
        model = models.resnet18(weights="DEFAULT")

        # Replace the final fully connected layer
        # ResNet18 structure: (fc): Linear(in_features=512, out_features=1000, bias=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load EfficientNet-B0 with default pre-trained weights
        model = models.efficientnet_b0(weights="DEFAULT")

        # Replace the final classification layer within the Sequential block
        # EfficientNet structure:
        # (classifier): Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        # We keep the dropout and replace the linear layer.
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        # Load DenseNet121 with default pre-trained weights
        model = models.densenet121(weights="DEFAULT")

        # Replace the classifier
        # DenseNet structure: (classifier): Linear(in_features=1024, out_features=1000, bias=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Architecture '{model_name}' not supported. "
            f"Choose from: resnet18, efficientnet_b0, densenet121"
        )

    # Move model to the specified device
    model = model.to(device)

    return model
