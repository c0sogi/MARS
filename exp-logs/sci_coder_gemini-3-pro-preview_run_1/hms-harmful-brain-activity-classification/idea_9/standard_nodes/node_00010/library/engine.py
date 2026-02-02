import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, kl_divergence_score, save_checkpoint


def train_one_epoch(dataloader, model, optimizer, scheduler, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    # KLDivLoss expects input to be log-probabilities and target to be probabilities
    criterion = nn.KLDivLoss(reduction="batchmean")

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        eeg = batch["eeg"].to(device, non_blocking=True)
        spec = batch["spec"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass: Model outputs Softmax probabilities
        preds = model(eeg, spec)

        # Calculate Loss
        # Clamp predictions to avoid log(0)
        preds_clamped = torch.clamp(preds, min=1e-7, max=1.0)
        loss = criterion(torch.log(preds_clamped), targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Scheduler step (OneCycleLR updates per step)
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        loss_meter.update(loss.item(), eeg.size(0))

    return loss_meter.avg


def validate(dataloader, model, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            spec = batch["spec"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward pass
            preds = model(eeg, spec)

            # Calculate Metric (KL Divergence)
            score = kl_divergence_score(preds, targets)
            loss_meter.update(score, eeg.size(0))

    return loss_meter.avg


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_dir,
    patience=3,
):
    """
    Orchestrates the training process including early stopping and checkpointing.
    """
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(
            train_loader, model, optimizer, scheduler, device, epoch
        )
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss = validate(val_loader, model, device)
        print(f"Validation Loss: {val_loss}")

        # Checkpointing and Early Stopping Logic
        is_best = val_loss < best_score
        if is_best:
            best_score = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save Checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_score": best_score,
            },
            is_best,
            checkpoint_dir,
        )

        # Early Stopping Check
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs. Best Score: {best_score}"
            )
            break

    return best_score


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    all_preds = []
    all_eeg_ids = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            spec = batch["spec"].to(device, non_blocking=True)
            eeg_ids = batch["eeg_id"]

            # Forward pass
            preds = model(eeg, spec)

            all_preds.append(preds.cpu().numpy())
            # Handle eeg_id being tensor or list
            if isinstance(eeg_ids, torch.Tensor):
                all_eeg_ids.extend(eeg_ids.numpy())
            else:
                all_eeg_ids.extend(eeg_ids)

    # Concatenate predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Create DataFrame
    # Columns must match the sample submission format
    df = pd.DataFrame(all_preds, columns=Config.CLASS_NAMES)
    df.insert(0, "eeg_id", all_eeg_ids)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
