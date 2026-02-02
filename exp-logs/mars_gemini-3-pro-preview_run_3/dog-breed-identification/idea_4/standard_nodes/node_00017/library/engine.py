import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import AverageMeter, calculate_log_loss, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device to train on.
        epoch: Current epoch number (for display).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Define Loss function
    # Using label_smoothing from Config (default 0.0 for strict Log Loss optimization)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    scaler = GradScaler()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Training Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Log Loss.

    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        device: Device to evaluate on.

    Returns:
        float: Log Loss value.
    """
    model.eval()

    # Lists to store predictions and true labels for Log Loss calculation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Calculate Log Loss
    # We use the utility function which wraps sklearn.metrics.log_loss
    val_loss = calculate_log_loss(y_true, y_pred)

    # Print full precision as requested
    print(f"Validation Log Loss: {val_loss}")
    return val_loss


def predict_tta(model, dataloader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model.
        dataloader: Test dataloader (returns image, id).
        device: Device to predict on.

    Returns:
        tuple: (list of ids, numpy array of predictions)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device, non_blocking=True)

            with autocast():
                # 1. Forward pass with original images
                outputs_orig = model(images)
                probs_orig = torch.softmax(outputs_orig, dim=1)

                # 2. Forward pass with horizontally flipped images
                # Flip along width dimension (dim 3 for NCHW tensor)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # 3. Average predictions
                avg_probs = (probs_orig + probs_flipped) / 2.0

            all_preds.append(avg_probs.float().cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions
    predictions = np.concatenate(all_preds)
    return all_ids, predictions


def train_phase(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    patience,
    save_name="checkpoint.pth",
):
    """
    Runs a training phase (loop over epochs) with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs: Maximum number of epochs.
        device: Device.
        patience: Early stopping patience.
        save_name: Filename for saving the best model.

    Returns:
        float: Best validation loss achieved.
    """
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting Epoch {epoch}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Early Stopping & Checkpointing
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            print(f"New best model found! Loss: {best_loss}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_loss": best_loss,
            },
            is_best,
            filename=save_name,
            output_dir=Config.OUTPUT_DIR,
        )

        if patience_counter >= patience:
            print("Early Stopping triggered.")
            break

    return best_loss


def generate_submission(model, test_loader, classes, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: PyTorch model.
        test_loader: Test dataloader.
        classes: List of class names (column headers).
        device: Device.
        output_path: Path to save the submission CSV.
    """
    print("Generating submission with TTA...")
    ids, preds = predict_tta(model, test_loader, device)

    # Create DataFrame
    df_sub = pd.DataFrame(preds, columns=classes)
    df_sub.insert(0, "id", ids)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
