import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    grad_accum_steps: int = 1,
) -> float:
    """
    Trains the model for one epoch using Gradient Accumulation.

    Args:
        model: The neural network.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device to train on.
        criterion: Loss function.
        grad_accum_steps: Number of steps to accumulate gradients before updating.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate unscaled loss for reporting
        running_loss += (loss.item() * grad_accum_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network.
        dataloader: Validation dataloader.
        device: Device to evaluate on.
        criterion: Loss function.

    Returns:
        tuple: (Average Loss, Macro F1 Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate Macro F1 Score
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    # Print full precision metrics as requested
    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Macro F1: {macro_f1}")

    return avg_loss, macro_f1


def configure_model_for_stage(model: nn.Module, stage: int):
    """
    Configures the model's frozen/unfrozen layers based on the training stage.

    Stage 1: Freeze backbone, train only classifier.
    Stage 2: Unfreeze top blocks of backbone, train classifier + top blocks.
    """
    if stage == 1:
        # Freeze all backbone layers
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        print("Stage 1 Configuration: Backbone FROZEN, Classifier UNFROZEN.")

    elif stage == 2:
        # Unfreeze specific top blocks of the backbone
        # efficientnet_b4.features is a Sequential container.
        # We unfreeze the last 3 blocks (indices -1, -2, -3) to allow high-level feature adaptation.

        # First, ensure everything is frozen initially to be safe
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 3 blocks of the features
        # Note: Accessing children of Sequential by slicing or iteration
        total_blocks = len(model.backbone)
        blocks_to_unfreeze = 3
        start_idx = max(0, total_blocks - blocks_to_unfreeze)

        for i in range(start_idx, total_blocks):
            for param in model.backbone[i].parameters():
                param.requires_grad = True

        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

        print(
            f"Stage 2 Configuration: Backbone blocks {start_idx}-{total_blocks-1} UNFROZEN, Classifier UNFROZEN."
        )


def get_optimizer(model: nn.Module, stage: int) -> torch.optim.Optimizer:
    """
    Returns the optimizer for the specific training stage.
    """
    # Filter parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if stage == 1:
        # Stage 1: Adam, higher LR, no weight decay usually for head init
        return torch.optim.Adam(
            trainable_params,
            lr=Config.LR_STAGE1,
            weight_decay=0,  # Often 0 for initial head alignment
        )
    elif stage == 2:
        # Stage 2: AdamW, lower LR, weight decay for regularization
        return torch.optim.AdamW(
            trainable_params, lr=Config.LR_STAGE2, weight_decay=Config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def get_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    """
    Returns a Cosine Annealing scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
