import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import AverageMeter, quadratic_weighted_kappa


def train_one_epoch(model, dataloader, optimizer, device, epoch, accumulation_steps=1):
    """
    Trains the model for one epoch using Mean Squared Error loss and Gradient Accumulation.

    Args:
        model: PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device to train on.
        epoch: Current epoch number.
        accumulation_steps: Number of steps to accumulate gradients before updating.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    criterion = nn.MSELoss()
    num_steps = len(dataloader)

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        # Ensure targets are float and have shape (B, 1) for regression
        targets = targets.to(device).view(-1, 1)

        outputs = model(images)
        loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every 'accumulation_steps' or at the end of the epoch
        if ((step + 1) % accumulation_steps == 0) or ((step + 1) == num_steps):
            optimizer.step()
            optimizer.zero_grad()

        # Update meter with the actual loss value (unscaled)
        loss_meter.update(loss.item() * accumulation_steps, images.size(0))

    print(f"Epoch {epoch} Training Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        device: Device to evaluate on.

    Returns:
        tuple: (kappa_score, average_loss)
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.MSELoss()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    # Concatenate predictions and targets
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    # Post-process predictions for QWK calculation
    # Clip to [0, 4] and round to nearest integer
    preds_processed = np.round(np.clip(preds, 0, 4)).astype(int)
    targets_int = targets.astype(int)

    kappa = quadratic_weighted_kappa(targets_int, preds_processed)

    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation Kappa: {kappa}")

    return kappa, loss_meter.avg


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    accumulation_steps=1,
    patience=5,
    save_path="./working/idea_5/best_model.pth",
    scheduler=None,
):
    """
    Executes the training loop with early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        device: Device.
        epochs: Maximum number of epochs.
        accumulation_steps: Gradient accumulation steps.
        patience: Early stopping patience.
        save_path: Path to save the best model.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Best validation kappa score.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_kappa = -float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, accumulation_steps
        )
        val_kappa, val_loss = validate(model, val_loader, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early Stopping Logic
        if val_kappa > best_kappa:
            print(f"Kappa improved from {best_kappa} to {val_kappa}. Saving model...")
            best_kappa = val_kappa
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Kappa did not improve. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Kappa: {best_kappa}")
    return best_kappa


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: PyTorch model (should be loaded with best weights).
        test_loader: Test dataloader.
        device: Device.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    preds_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds_list.append(outputs.cpu().numpy())

    preds = np.concatenate(preds_list)

    # Post-process: Clip to [0, 4] and round
    preds_final = np.round(np.clip(preds, 0, 4)).astype(int).flatten()

    # Load sample submission to ensure correct format and IDs
    sample_sub_path = "./input/sample_submission.csv"
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Sample submission not found at {sample_sub_path}")

    sample_sub = pd.read_csv(sample_sub_path)

    # Assign predictions
    # Note: This assumes the test_loader yields images in the same order as sample_submission.csv
    if len(preds_final) != len(sample_sub):
        print(
            f"Warning: Number of predictions ({len(preds_final)}) does not match sample submission ({len(sample_sub)})."
        )

    sample_sub["diagnosis"] = preds_final

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
