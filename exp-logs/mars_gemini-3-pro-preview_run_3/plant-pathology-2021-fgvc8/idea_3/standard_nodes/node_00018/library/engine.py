import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import calculate_f1_score


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns: (val_loss, val_f1)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    val_loss = running_loss / dataset_size

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate F1 Score
    val_f1 = calculate_f1_score(
        all_logits, all_targets, threshold=Config.CONF_THRESHOLD
    )

    return val_loss, val_f1


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=Config.EPOCHS,
    scheduler=None,
    patience=5,
):
    """
    Orchestrates the training process with Early Stopping and Model Checkpointing.
    """
    scaler = GradScaler()
    best_f1 = -1.0
    best_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(num_epochs):
        # Training Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validation Step
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val F1: {val_f1}"
        )

        # Checkpointing & Early Stopping
        # We prioritize F1 score for improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"Validation F1 improved. Model saved to {Config.BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in F1 for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training complete. Best Val F1: {best_f1}")


def predict_with_tta(model, loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions using Test Time Augmentation (TTA) and saves to CSV.
    TTA Strategy: Average probabilities of Original, Horizontal Flip, and Vertical Flip.
    """
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Best Model
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    model.to(device)

    all_preds = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is width in NCHW)
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is height in NCHW)
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)

            # Average Probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Move to CPU
            all_preds.append(avg_probs.cpu().numpy())

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Thresholding
    binary_preds = (all_preds > Config.CONF_THRESHOLD).astype(int)

    # Map to Labels
    # Config.LABELS is sorted alphabetically
    idx_to_label = {i: label for i, label in enumerate(Config.LABELS)}

    submission_rows = []

    # Get image IDs from the dataset dataframe
    # Note: We assume the loader preserves order (shuffle=False for test)
    image_ids = loader.dataset.df["image"].values

    for i, row_binary in enumerate(binary_preds):
        image_id = image_ids[i]

        # Get list of predicted labels
        predicted_indices = np.where(row_binary == 1)[0]
        predicted_labels = [idx_to_label[idx] for idx in predicted_indices]

        # Join with space
        label_str = " ".join(predicted_labels)

        submission_rows.append({"image": image_id, "labels": label_str})

    # Create DataFrame and Save
    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}. Total predictions: {len(df_submission)}")
