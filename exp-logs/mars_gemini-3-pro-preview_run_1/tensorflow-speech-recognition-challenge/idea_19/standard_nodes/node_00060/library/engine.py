import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import get_logger, get_fine_grained_labels
from library.sam import SAM
from library.model import DilatedEfficientNet
from library.dataset import get_dataloader


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using SAM and Mixup.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    logger = get_logger()

    for i, (inputs, targets, _) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # -------------------------------------------------------
        # Mixup Preparation
        # -------------------------------------------------------
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, Config.MIXUP_ALPHA, device
        )

        # -------------------------------------------------------
        # SAM Step 1: Ascent (Find worst case in neighborhood)
        # -------------------------------------------------------
        # First Forward Pass
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # First Backward Pass
        loss.backward()

        # Ascent Step
        optimizer.first_step(zero_grad=True)

        # -------------------------------------------------------
        # SAM Step 2: Descent (Update weights minimizing worst case)
        # -------------------------------------------------------
        # Second Forward Pass (at perturbed weights)
        outputs_2 = model(inputs)
        loss_2 = mixup_criterion(criterion, outputs_2, targets_a, targets_b, lam)

        # Second Backward Pass
        loss_2.backward()

        # Descent Step
        optimizer.second_step(zero_grad=True)

        # -------------------------------------------------------
        # Metrics
        # -------------------------------------------------------
        running_loss += loss.item() * inputs.size(0)

        # Accuracy (approximation for Mixup: compare to the dominant label)
        _, predicted = outputs.max(1)
        # We count correct if it matches either label, weighted?
        # Standard practice: just check against the original target (y_a) if lam > 0.5 else y_b
        # Or simpler: just track loss primarily. For display, we compare to y_a.
        total += targets.size(0)
        correct += (
            predicted.eq(targets_a).sum().item()
            if lam >= 0.5
            else predicted.eq(targets_b).sum().item()
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    logger.info(f"Epoch {epoch} [Train] Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.6f}")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, epoch=None):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    logger = get_logger()

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    loss = running_loss / total
    acc = correct / total

    prefix = f"Epoch {epoch} " if epoch is not None else ""
    logger.info(f"{prefix}[Val]   Loss: {loss:.6f} | Acc: {acc:.6f}")

    return loss, acc


def train_model():
    """
    Main function to train the model.
    """
    logger = get_logger()
    logger.info("Starting training process...")

    # 1. Setup Device
    device = torch.device(Config.DEVICE)

    # 2. Prepare Data
    train_loader = get_dataloader(split="train", mode="train", shuffle=True)
    val_loader = get_dataloader(split="val", mode="infer", shuffle=False)

    # 3. Initialize Model
    fine_labels = get_fine_grained_labels()
    num_classes = len(fine_labels)
    logger.info(f"Initializing DilatedEfficientNet for {num_classes} classes.")

    model = DilatedEfficientNet(num_classes=num_classes)
    model.to(device)

    # 4. Optimizer (SAM wrapping AdamW)
    # Note: SAM takes the base optimizer class, not an instance
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=Config.SAM_RHO,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 5. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer, T_max=Config.EPOCHS
    )

    # 6. Loss Function
    criterion = nn.CrossEntropyLoss()

    # 7. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience = 10
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)

        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            logger.info(f"New best accuracy: {best_acc:.6f}. Saving model...")
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        duration = time.time() - start_time
        logger.info(
            f"Epoch {epoch} completed in {duration:.2f}s. LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Early Stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.6f}")
    return best_model_path
