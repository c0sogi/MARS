import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from library.config import Config


def get_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """
    Constructs the ConvNeXt Tiny model and adapts the classifier head for the task.

    Args:
        num_classes (int): Number of output classes (dog breeds).
        pretrained (bool): If True, loads ImageNet pretrained weights.

    Returns:
        nn.Module: The adapted ConvNeXt model.
    """
    # Load appropriate weights
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None

    # Initialize model
    model = convnext_tiny(weights=weights)

    # Modify the classifier head
    # The torchvision ConvNeXt classifier is a Sequential:
    # (0): LayerNorm2d
    # (1): Flatten
    # (2): Linear
    # We replace the last Linear layer to match our number of classes.

    original_head = model.classifier[2]
    in_features = original_head.in_features

    # Replace with new Linear layer
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT_RATE),
        nn.Linear(in_features, num_classes),
    )

    return model


def setup_phase(model, phase):
    """
    Configures parameter freezing and initializes the optimizer for the specific training phase.

    Args:
        model (nn.Module): The model to configure.
        phase (str): 'phase1' (Head Adaptation) or 'phase2' (Fine-Tuning).

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    if phase == "phase1":
        # Phase 1: Head Adaptation
        # Freeze the entire backbone, train only the classifier head.

        # 1. Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # 2. Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True

        # 3. Create Optimizer for trainable parameters only
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.PHASE1_LR,
            weight_decay=Config.WEIGHT_DECAY,
        )

    elif phase == "phase2":
        # Phase 2: Fine-Tuning with Discriminative Learning Rates
        # Unfreeze the last stage of the backbone (Stage 4) and the head.

        # 1. Freeze all parameters (reset state)
        for param in model.parameters():
            param.requires_grad = False

        # 2. Unfreeze Stage 4 of the backbone
        # In torchvision's convnext_tiny, 'features' is a Sequential where index 7 is Stage 4.
        for param in model.features[7].parameters():
            param.requires_grad = True

        # 3. Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True

        # 4. Create Optimizer with parameter groups for discriminative LRs
        param_groups = [
            {"params": model.features[7].parameters(), "lr": Config.PHASE2_BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": Config.PHASE2_HEAD_LR},
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    else:
        raise ValueError(f"Invalid phase '{phase}'. Options are 'phase1', 'phase2'.")

    return optimizer
