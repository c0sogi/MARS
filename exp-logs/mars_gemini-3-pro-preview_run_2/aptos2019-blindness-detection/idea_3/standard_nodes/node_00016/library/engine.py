import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, quadratic_weighted_kappa


def train_one_epoch(epoch, model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to compute on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Reshape outputs to match targets (Batch Size,)
        outputs = outputs.view(-1)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Train Epoch {epoch}: Loss = {loss_meter.avg}")
    return loss_meter.avg


def validate(epoch, model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to compute on.

    Returns:
        tuple: (Average Loss, Quadratic Weighted Kappa Score)
    """
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), images.size(0))

            # Store predictions and targets for metric calculation
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Convert regression outputs to ordinal labels
    # 1. Round to nearest integer
    # 2. Clip to valid range [0, 4]
    preds_rounded = np.round(all_preds)
    preds_clipped = np.clip(preds_rounded, 0, 4).astype(int)
    targets_int = all_targets.astype(int)

    # Calculate Quadratic Weighted Kappa
    kappa = quadratic_weighted_kappa(targets_int, preds_clipped)

    print(f"Validation Epoch {epoch}: Loss = {loss_meter.avg}, Kappa = {kappa}")

    return loss_meter.avg, kappa


def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
):
    """
    Runs the full training loop with Early Stopping and Model Checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device to compute on.
        epochs (int): Total number of epochs.
        save_path (str): Path to save the best model weights.
    """
    best_kappa = -float("inf")
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_kappa = validate(epoch, model, val_loader, criterion, device)

        # Step Scheduler (CosineAnnealing is usually stepped per epoch)
        if scheduler is not None:
            scheduler.step()

        # Model Checkpointing
        if val_kappa > best_kappa:
            print(
                f"Kappa improved from {best_kappa} to {val_kappa}. Saving model to {save_path}"
            )
            best_kappa = val_kappa
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Kappa did not improve. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best Kappa: {best_kappa}")


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to compute on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    all_preds = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(-1)
            all_preds.append(outputs.cpu().numpy())

    # Process predictions
    all_preds = np.concatenate(all_preds)
    preds_rounded = np.round(all_preds)
    preds_clipped = np.clip(preds_rounded, 0, 4).astype(int)

    # Load test metadata to get ID codes
    # We use the metadata file because the loader only yields images in test mode
    try:
        df_test = pd.read_csv(Config.test_csv_path)
    except FileNotFoundError:
        # Fallback to sample submission if metadata is missing (unlikely)
        df_test = pd.read_csv(Config.sample_submission_path)

    # Ensure alignment
    if len(df_test) != len(preds_clipped):
        print(
            f"Warning: Number of predictions ({len(preds_clipped)}) does not match number of test samples ({len(df_test)})."
        )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"id_code": df_test["id_code"], "diagnosis": preds_clipped}
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
