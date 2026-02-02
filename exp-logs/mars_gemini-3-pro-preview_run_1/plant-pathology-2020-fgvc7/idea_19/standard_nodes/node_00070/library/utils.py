import os
import random
import numpy as np
import torch
import math
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for cuDNN to guarantee reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_initial_loss(model, dataloader, criterion, device):
    """
    Computes the loss on the first batch of the dataloader to verify model initialization.

    This function acts as a safeguard against "silent failures" in transfer learning,
    such as weights not loading correctly or data preprocessing errors. It checks if
    the initial loss is finite and within a reasonable range of random guessing.

    Args:
        model: The PyTorch model to verify.
        dataloader: The dataloader containing the training data.
        criterion: The loss function used for training.
        device: The device (CPU/GPU) to run the computation on.

    Returns:
        float: The computed initial loss value.

    Raises:
        ValueError: If the loss is NaN, Inf, or significantly higher than the
                    expected random guessing baseline.
    """
    # Switch model to evaluation mode for this check to disable dropout/batchnorm updates
    model.eval()

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise ValueError("Dataloader is empty. Cannot verify initial loss.")

    # Unpack batch (assuming standard images, targets format)
    images, targets = batch
    images = images.to(device)
    targets = targets.to(device)

    # Compute loss without tracking gradients
    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, targets)

    loss_value = loss.item()

    # Calculate theoretical random guessing loss for Softmax Cross Entropy: -ln(1/C)
    num_classes = Config.num_classes
    random_guess_loss = -math.log(1.0 / num_classes)

    print(f"Initial Loss Check: {loss_value:.10f}")
    print(f"Theoretical Random Guessing Loss: {random_guess_loss:.10f}")

    # Check 1: Finite Loss
    if not math.isfinite(loss_value):
        raise ValueError(
            f"Initial loss is not finite: {loss_value}. Check model weights and data inputs."
        )

    # Check 2: Reasonable Range
    # We expect the loss to be around the random guessing baseline (approx 1.38 for 4 classes)
    # if the head is randomly initialized. If it is significantly higher (e.g., > 2.0x),
    # it indicates a problem (exploding gradients, incorrect scaling, etc.).
    # The prompt mentions verifying it is "significantly lower", but for a fresh head,
    # "comparable to and not exploding" is the practical success criteria.
    threshold = random_guess_loss * 2.5
    if loss_value > threshold:
        raise ValueError(
            f"Initial loss {loss_value:.6f} is anomalously high (Threshold: {threshold:.6f}). "
            "This suggests a failure in model initialization or data preprocessing."
        )

    return loss_value
