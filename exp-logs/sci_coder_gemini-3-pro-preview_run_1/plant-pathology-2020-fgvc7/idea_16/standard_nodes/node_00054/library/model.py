import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import numpy as np
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="model")


class AppleResNet34(nn.Module):
    """
    ResNet34-based model for Apple Disease Detection.
    Replaces the final fully connected layer to output probabilities for 4 classes.
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights.
        """
        super(AppleResNet34, self).__init__()

        # Determine weights
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None

        # Load backbone
        self.backbone = models.resnet34(weights=weights)

        # Replace the final fully connected layer
        # ResNet34 fc layer: (avgpool): AdaptiveAvgPool2d(output_size=(1, 1)) -> (fc): Linear(in_features=512, ...)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

        logger.info(f"Initialized AppleResNet34 (Pretrained={pretrained})")

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].
        Returns:
            torch.Tensor: Logits [B, NUM_CLASSES].
        """
        return self.backbone(x)


def verify_initial_loss(model, loader, criterion, device):
    """
    Passes one batch through the model to verify the loss is reasonable.
    This acts as a safeguard against silent failures (e.g., data pipeline issues,
    broken weights) before expensive training begins.

    Args:
        model (nn.Module): The model to verify.
        loader (DataLoader): DataLoader to fetch a single batch from.
        criterion (loss_fn): The loss function.
        device (torch.device): Device to run the check on.

    Raises:
        RuntimeError: If the initial loss is suspiciously high.
    """
    model.eval()
    model.to(device)

    logger.info("Running Initial Loss Test...")

    try:
        # Fetch one batch
        images, targets = next(iter(loader))
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass (no grad)
        with torch.no_grad():
            outputs = model(images)
            # Check if targets are one-hot or indices for CrossEntropyLoss
            # Config.TARGET_COLS implies multi-class (4 classes).
            # If targets are probabilities (float), CrossEntropyLoss expects class indices
            # OR probabilities (if supported by version, usually requires indices for older torch).
            # However, dataset returns float32. We usually take argmax for CE if not using BCEWithLogits.
            # Config says LOSS_FUNCTION = "CrossEntropyLoss".
            # Standard CrossEntropyLoss in PyTorch expects class indices (Long) for target,
            # or probabilities (Float) if using soft labels (available in newer PyTorch).
            # Given the dataset returns float32 one-hot-like/distribution, we should check.

            # If targets are float (probabilities), we might need to convert to indices for standard CE
            # or keep as is if using soft target support.
            # Assuming standard usage:
            if targets.dtype == torch.float32 and len(targets.shape) > 1:
                # If using standard CE, usually expects indices.
                # But let's assume the training loop handles this or the loss function handles soft labels.
                # For the check, we'll calculate loss as is.
                pass

            loss = criterion(outputs, targets)

        loss_val = loss.item()
        logger.info(f"Initial Batch Loss: {loss_val:.4f}")

        # Random guessing loss for 4 classes is -ln(0.25) ~= 1.386
        # We allow a margin for random initialization variance.
        # If loss is > 2.5, something is likely wrong (e.g. unnormalized data).
        threshold = 2.5
        if loss_val > threshold:
            msg = (
                f"Initial loss {loss_val:.4f} exceeds safety threshold {threshold}. "
                "Check data normalization or model initialization."
            )
            logger.warning(msg)
            # We raise an error to strictly enforce the 'significantly lower' or 'reasonable' requirement
            # preventing silent failure.
            raise RuntimeError(msg)

        logger.info("Initial Loss Test Passed.")

    except Exception as e:
        logger.error(f"Initial Loss Test Failed: {e}")
        raise e
    finally:
        model.train()
