import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import MetricMonitor, quadratic_weighted_kappa


def train_fn(model, train_loader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The PyTorch model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        device (torch.device): Device to run training on (CPU/GPU).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_monitor = MetricMonitor()
    criterion = nn.MSELoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        loss_monitor.update(loss.item(), images.size(0))

    return loss_monitor.avg


def eval_fn(model, val_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The PyTorch model.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, qwk_score)
    """
    model.eval()
    loss_monitor = MetricMonitor()
    criterion = nn.MSELoss()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_monitor.update(loss.item(), images.size(0))

            # Store predictions and targets for QWK calculation
            final_targets.extend(targets.cpu().numpy().tolist())
            final_outputs.extend(outputs.cpu().numpy().tolist())

    # Convert to numpy arrays
    predictions = np.array(final_outputs)
    targets = np.array(final_targets)

    # Post-processing for Regression to Classification
    # 1. Clip values to valid range [0, 4]
    predictions = np.clip(predictions, 0, 4)
    # 2. Round to nearest integer
    predictions = np.round(predictions).astype(int)
    targets = targets.astype(int)

    # Calculate Quadratic Weighted Kappa
    qwk = quadratic_weighted_kappa(targets, predictions)

    return loss_monitor.avg, qwk


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=5,
):
    """
    Runs the full training loop with early stopping and scheduler.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
    """
    best_kappa = -float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, device)

        # Step the scheduler (Cite solution_lesson_node_00003)
        if scheduler is not None:
            scheduler.step()

        # Evaluate
        val_loss, val_kappa = eval_fn(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{num_epochs} | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Kappa: {val_kappa}")

        # Early Stopping and Model Checkpointing
        # We maximize Kappa
        if val_kappa > best_kappa:
            print(
                f"Validation Kappa improved from {best_kappa} to {val_kappa}. Saving model..."
            )
            best_kappa = val_kappa
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in Kappa. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Kappa: {best_kappa}")
