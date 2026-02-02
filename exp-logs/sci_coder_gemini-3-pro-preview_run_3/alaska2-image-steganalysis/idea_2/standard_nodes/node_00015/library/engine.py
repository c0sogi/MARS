import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.model import MonoResidualEfficientNet, BCEWithLogitsLossWithSmoothing
from library.dataset import get_dataloaders, get_test_dataloader
from library.metrics import alaska_weighted_auc


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        # Flatten logits to match label shape (N,)
        loss = criterion(logits.view(-1), labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)
            dataset_size += batch_size

            logits = model(images)
            loss = criterion(logits.view(-1), labels)

            running_loss += loss.item() * batch_size

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(logits).view(-1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Weighted AUC
    val_score = alaska_weighted_auc(all_labels, all_preds)

    return val_loss, val_score


def predict_tta(model, loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (4-view rotation).
    """
    model.eval()

    # Ensure we get IDs. The loader returns (image, label), where label is dummy.
    ids_list = loader.dataset.df["image_id"].values
    preds_list = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)

            # TTA Strategy: Average predictions of 0, 90, 180, 270 degree rotations

            # View 1: Original
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1).view(-1)

            # View 2: 90 degrees
            # Rot90 rotates in the plane defined by dims (2, 3) for (N, C, H, W)
            images_90 = torch.rot90(images, k=1, dims=[2, 3])
            logits_2 = model(images_90)
            probs_2 = torch.sigmoid(logits_2).view(-1)

            # View 3: 180 degrees
            images_180 = torch.rot90(images, k=2, dims=[2, 3])
            logits_3 = model(images_180)
            probs_3 = torch.sigmoid(logits_3).view(-1)

            # View 4: 270 degrees
            images_270 = torch.rot90(images, k=3, dims=[2, 3])
            logits_4 = model(images_270)
            probs_4 = torch.sigmoid(logits_4).view(-1)

            # Average probabilities
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            preds_list.extend(avg_probs.cpu().numpy())

    return ids_list, np.array(preds_list)


def run_training(debug=False):
    """
    Main execution function for training and inference.

    Args:
        debug (bool): If True, runs on a small subset of data for fewer epochs.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Starting run_training. Device: {device}, Debug: {debug}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(Config.CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loaders
    train_loader, val_loader = get_dataloaders(debug=debug)

    # 3. Model
    model = MonoResidualEfficientNet(backbone_name=Config.BACKBONE, pretrained=True)
    model = model.to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    epochs = 2 if debug else Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss: BCEWithLogitsLoss with Label Smoothing
    criterion = BCEWithLogitsLossWithSmoothing(label_smoothing=Config.LABEL_SMOOTHING)

    # 5. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Val Weighted AUC: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            print(
                f"New best score found ({val_score}). Saving model to {Config.CHECKPOINT_PATH}"
            )
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Score did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Starting inference on Test set...")

    # Load best model
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    test_loader = get_test_dataloader()

    if test_loader is not None:
        ids, preds = predict_tta(model, test_loader, device)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"Id": ids, "Label": preds})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
    else:
        print("Test loader is None. Skipping inference.")
