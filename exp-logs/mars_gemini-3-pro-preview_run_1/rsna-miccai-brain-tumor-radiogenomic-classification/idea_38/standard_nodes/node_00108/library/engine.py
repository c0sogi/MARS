import torch
import numpy as np
import pandas as pd
import os
from library.utils import AverageMeter, get_score, save_checkpoint, print_metrics
from library.config import (
    MODEL_SAVE_PATH,
    EARLY_STOPPING_PATIENCE,
    NUM_EPOCHS,
    SUBMISSION_PATH,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in dataloader:
        # Move data to device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        # labels are (B,), logits are (B, 1). Unsqueeze labels to match.
        loss = criterion(logits, labels.unsqueeze(1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    Performs evaluation on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels.unsqueeze(1))

            losses.update(loss.item(), images.size(0))

            # Convert logits to probabilities
            preds = torch.sigmoid(logits)

            # Store predictions and labels for AUC calculation
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate ROC AUC
    score = get_score(all_labels, all_preds)

    return losses.avg, score


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=NUM_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
):
    """
    Runs the full training loop with Early Stopping.
    """
    best_score = 0.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_score = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Print metrics
        print_metrics(epoch, train_loss, val_loss, val_score)

        # Early Stopping and Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0

            # Save best model
            checkpoint = {
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "epoch": epoch,
            }
            save_checkpoint(checkpoint, MODEL_SAVE_PATH)
            print(f"Validation score improved. Model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    return best_score


def generate_submission(model, test_loader, device, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            # BraTS21ID is a tensor of IDs
            ids = batch["BraTS21ID"]

            logits = model(images)
            preds = torch.sigmoid(logits)

            ids_list.extend(ids.numpy().flatten())
            preds_list.extend(preds.cpu().numpy().flatten())

    # Create submission DataFrame
    df = pd.DataFrame({"BraTS21ID": ids_list, "MGMT_value": preds_list})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
