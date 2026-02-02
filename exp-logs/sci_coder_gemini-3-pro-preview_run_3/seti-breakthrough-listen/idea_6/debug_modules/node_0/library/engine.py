import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    AverageMeter,
    get_score,
    mixup_data,
    mixup_criterion,
    seed_everything,
)
from library.dataset import get_train_val_loaders, get_test_loader
from library.model import SiameseDifferenceNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()

    for on_imgs, off_imgs, targets in loader:
        on_imgs = on_imgs.to(device)
        off_imgs = off_imgs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        # Note: We mix both on_imgs and off_imgs with the same lambda/indices
        # to preserve the relationship between the streams.
        mixed_on, mixed_off, y_a, y_b, lam = mixup_data(
            on_imgs, off_imgs, targets, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        preds = model(mixed_on, mixed_off).squeeze(1)

        # Calculate Mixup Loss
        loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), on_imgs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for on_imgs, off_imgs, targets in loader:
            on_imgs = on_imgs.to(device)
            off_imgs = off_imgs.to(device)
            targets = targets.to(device)

            # Forward pass (No Mixup)
            logits = model(on_imgs, off_imgs).squeeze(1)
            loss = criterion(logits, targets)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), on_imgs.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    auc = get_score(all_targets, all_preds)
    return losses.avg, auc


def run_training(
    debug=Config.DEBUG,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    save_path=Config.BEST_MODEL_PATH,
):
    """
    Main training orchestration function.
    Handles data loading, model initialization, training loop,
    early stopping, and model saving.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader = get_train_val_loaders(debug=debug)

    # Model
    model = SiameseDifferenceNet().to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New Best AUC! Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def predict(model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set using the best saved model.
    Applies Test Time Augmentation (TTA).
    Saves results to submission.csv.
    """
    device = Config.DEVICE
    print("\nStarting inference...")

    # Load Metadata to ensure ID alignment
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_test = pd.read_csv(Config.TEST_METADATA)

    # Load Data
    test_loader = get_test_loader()

    # Load Model
    model = SiameseDifferenceNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Run training first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for on_imgs, off_imgs, _ in test_loader:
            on_imgs = on_imgs.to(device)
            off_imgs = off_imgs.to(device)

            # TTA Strategy:
            # 1. Original Input
            logits_orig = model(on_imgs, off_imgs).squeeze(1)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Flipped Input (Horizontal + Vertical)
            # Dims are (B, C, H, W). Flip H (dim 2) and W (dim 3).
            # This corresponds to Frequency Inversion + Time Reversal.
            on_imgs_flip = torch.flip(on_imgs, [2, 3])
            off_imgs_flip = torch.flip(off_imgs, [2, 3])

            logits_flip = model(on_imgs_flip, off_imgs_flip).squeeze(1)
            probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0
            all_preds.extend(avg_probs.cpu().numpy())

    # Save Submission
    # Ensure length matches
    if len(all_preds) != len(df_test):
        print(
            f"Warning: Prediction count ({len(all_preds)}) does not match metadata count ({len(df_test)})."
        )

    df_test["target"] = all_preds

    # Select only required columns
    submission_df = df_test[["id", "target"]]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(submission_df.head())
