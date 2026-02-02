import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.model import AppleDiseaseModel
from library.dataset import get_dataloaders
from library.utils import seed_everything


def reconstruct_probabilities(probs_2d: np.ndarray) -> np.ndarray:
    """
    Reconstructs 4-class probabilities from 2-class (Rust, Scab) probabilities
    based on the decomposition logic.

    Args:
        probs_2d (np.ndarray): Shape (N, 2). Col 0: Rust Prob, Col 1: Scab Prob.

    Returns:
        np.ndarray: Shape (N, 4). Columns: Healthy, Multiple, Rust, Scab.
    """
    pr = probs_2d[:, 0]
    ps = probs_2d[:, 1]

    # Decomposition Logic:
    # Healthy: No Rust AND No Scab
    healthy = (1 - pr) * (1 - ps)
    # Multiple: Rust AND Scab
    multiple = pr * ps
    # Rust: Rust AND No Scab (Exclusive Rust)
    rust = pr * (1 - ps)
    # Scab: No Rust AND Scab (Exclusive Scab)
    scab = (1 - pr) * ps

    # Stack in the order required for metric calculation and submission
    # Order: Healthy, Multiple, Rust, Scab
    return np.stack([healthy, multiple, rust, scab], axis=1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    # Progress bar
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # Apply Label Smoothing manually for BCE
        # y_smooth = y * (1 - alpha) + 0.5 * alpha
        # 0 -> 0.025, 1 -> 0.975 (given alpha=0.05)
        targets_smooth = (
            targets * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
        )

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets_smooth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Mean Column-wise ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            # For validation loss, we typically compare against raw targets
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds_2d = np.concatenate(all_preds)
    all_targets_2d = np.concatenate(all_targets)

    # Reconstruct 4-class probabilities for metric calculation
    # Since targets are binary 0/1, the reconstruction logic works perfectly to restore the one-hot encoding
    pred_4c = reconstruct_probabilities(all_preds_2d)
    target_4c = reconstruct_probabilities(all_targets_2d)

    # Calculate Mean Column-wise ROC AUC
    # We use 'macro' average to treat all classes equally
    try:
        auc_score = roc_auc_score(target_4c, pred_4c, average="macro")
    except ValueError:
        # Fallback in case of degenerate single-class batches (unlikely in full validation)
        auc_score = 0.5

    return epoch_loss, auc_score


def train_model():
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Data
    print("Loading Data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Initialize Model
    print(f"Initializing {Config.MODEL_NAME}...")
    model = AppleDiseaseModel(pretrained=True)
    model.to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Scheduler (Cite Lesson 00014)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc:.15f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def predict_and_submit():
    """
    Inference routine.
    Loads the best model, predicts on test set with TTA, and saves submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model = AppleDiseaseModel(pretrained=False)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    results = []
    print(f"Starting inference with TTA ({Config.TTA_STEPS} steps)...")

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)

            # TTA Step 1: Original Images
            logits1 = model(images)
            probs1 = torch.sigmoid(logits1)

            # TTA Step 2: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            logits2 = model(images_flipped)
            probs2 = torch.sigmoid(logits2)

            # Average Probabilities
            avg_probs = (probs1 + probs2) / 2.0
            avg_probs_np = avg_probs.cpu().numpy()

            # Reconstruct 4-class probabilities
            # Returns shape (B, 4): [Healthy, Multiple, Rust, Scab]
            final_probs = reconstruct_probabilities(avg_probs_np)

            # Store results
            for i, img_id in enumerate(image_ids):
                results.append(
                    {
                        "image_id": img_id,
                        "healthy": final_probs[i, 0],
                        "multiple_diseases": final_probs[i, 1],
                        "rust": final_probs[i, 2],
                        "scab": final_probs[i, 3],
                    }
                )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(submission_df.head())
