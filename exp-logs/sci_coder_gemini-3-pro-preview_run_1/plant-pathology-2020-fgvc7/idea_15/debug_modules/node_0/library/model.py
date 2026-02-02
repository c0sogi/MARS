import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from library.config import CFG


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.
    Uses a pre-trained backbone with a custom head for 4-class classification.
    """

    def __init__(self, pretrained=True):
        super(AppleResNet34, self).__init__()

        # Load ResNet34 backbone
        # Handling potential API differences in torchvision versions
        try:
            from torchvision.models import ResNet34_Weights

            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        except (ImportError, AttributeError):
            self.backbone = models.resnet34(pretrained=pretrained)

        # Replace the final fully connected layer
        # ResNet34's fc layer: (512) -> (num_classes)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, CFG.num_classes)

        # Initialize the new head
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).
        Returns:
            torch.Tensor: Raw logits (B, num_classes).
        """
        return self.backbone(x)


def verify_initialization(model, dataloader, criterion, device):
    """
    Verifies that the model's initial loss is within a reasonable range.
    This acts as a safeguard against silent failures (e.g., broken data pipeline,
    incorrect weight loading) before commencing the full training loop.

    Args:
        model (nn.Module): The model to verify.
        dataloader (DataLoader): A dataloader to fetch a single batch from.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the check on.

    Raises:
        RuntimeError: If the initial loss is significantly higher than expected.
    """
    model.eval()
    model.to(device)

    print("\nRunning Initial Loss Test...")

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("Warning: Dataloader is empty. Skipping initialization verification.")
        return

    images, labels, _ = batch
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, labels)

    loss_val = loss.item()

    # Calculate Random Guessing Baseline
    # For 4 classes, uniform probability p=0.25 -> -ln(0.25) approx 1.386
    random_guessing_loss = -np.log(1.0 / CFG.num_classes)

    print(f"  Initial Loss: {loss_val:.6f}")
    print(f"  Random Guessing Baseline: {random_guessing_loss:.6f}")

    # Threshold check
    # We allow a small margin above random guessing to account for batch variance
    # and the random initialization of the head.
    # However, if loss is significantly higher (e.g., > 2.5), it indicates a problem.
    threshold = random_guessing_loss * 2.0

    if loss_val > threshold:
        raise RuntimeError(
            f"Initialization Verification Failed! \n"
            f"Initial loss ({loss_val:.4f}) is significantly higher than the "
            f"random guessing baseline ({random_guessing_loss:.4f}). \n"
            f"This suggests a failure in model initialization or data preprocessing. Aborting run."
        )

    print("  -> Initialization verified successfully. Starting training.\n")

    # Return model to training mode
    model.train()
