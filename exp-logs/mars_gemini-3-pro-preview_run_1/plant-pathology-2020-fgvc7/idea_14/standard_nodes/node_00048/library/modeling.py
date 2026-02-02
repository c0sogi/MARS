import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from library.config import Config


def get_model(device=Config.DEVICE, pretrained=Config.PRETRAINED):
    """
    Constructs the ResNet34 model with a custom classification head.

    To satisfy the strict initialization verification (Initial Loss < 1.38),
    this implementation initializes the final layer biases using the
    log-prior of the class distribution.

    Class Distribution (from metadata):
    - healthy: 374
    - multiple_diseases: 68
    - rust: 441
    - scab: 427
    - Total: 1310

    Args:
        device (torch.device): The device to place the model on.
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Returns:
        model (torch.nn.Module): The initialized model.
    """
    # Load ResNet34 backbone
    # Using pretrained=True for broad compatibility across torchvision versions
    model = models.resnet34(pretrained=pretrained)

    # Replace the fully connected layer
    # ResNet34 structure: ... -> AdaptiveAvgPool2d -> Flatten -> Linear (fc)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    # =========================================================================
    # Expert Initialization
    # =========================================================================
    # Calculate class probabilities based on training data analysis
    # Order: ["healthy", "multiple_diseases", "rust", "scab"]
    counts = np.array([374, 68, 441, 427], dtype=np.float32)
    probs = counts / np.sum(counts)

    # Calculate bias initialization: bias = log(probability)
    # This sets the initial model output to the class priors, minimizing initial loss
    # Expected Initial Loss: -sum(p * log(p)) approx 1.18
    bias_values = np.log(probs + 1e-7)  # epsilon for safety

    # Apply to model
    model.fc.bias.data = torch.tensor(bias_values, dtype=torch.float32)

    # Scale down the weights of the final layer to ensure the bias (priors) dominates
    # the initial predictions, rather than random noise from the weights.
    nn.init.xavier_uniform_(model.fc.weight)
    model.fc.weight.data.mul_(0.01)

    # Move model to the specified device
    model = model.to(device)

    return model


def verify_initialization(model, data_loader, criterion, device=Config.DEVICE):
    """
    Verifies that the model's initial loss is within the safe range defined in Config.

    This function acts as a safeguard against:
    1. Incorrect data normalization (causing massive activations).
    2. Broken transfer learning weights.
    3. Incorrect head initialization.

    If the loss is > Config.INIT_LOSS_THRESHOLD (1.38), it implies the model
    is performing worse than random guessing (or a uniform prior), which is
    unacceptable for a calibrated training run.

    Args:
        model (torch.nn.Module): The model to verify.
        data_loader (torch.utils.data.DataLoader): DataLoader to fetch a single batch.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): Device for computation.

    Returns:
        bool: True if verification passes.

    Raises:
        RuntimeError: If the initial loss exceeds the threshold.
    """
    print("Running Initialization Verification...")

    model.eval()

    try:
        # Fetch a single batch
        images, labels = next(iter(data_loader))
    except StopIteration:
        print("Warning: DataLoader is empty. Skipping verification.")
        return True

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, labels)

    initial_loss = loss.item()
    print(f"  Initial Batch Loss: {initial_loss:.6f}")
    print(f"  Threshold: {Config.INIT_LOSS_THRESHOLD}")

    # Restore training mode
    model.train()

    if initial_loss > Config.INIT_LOSS_THRESHOLD:
        raise RuntimeError(
            f"Initialization Verification Failed! "
            f"Initial loss ({initial_loss:.4f}) exceeds the safety threshold ({Config.INIT_LOSS_THRESHOLD}). "
            f"This indicates that the model head is not correctly initialized to class priors "
            f"or there is a data preprocessing issue."
        )

    print("Initialization Verification Passed.")
    return True
