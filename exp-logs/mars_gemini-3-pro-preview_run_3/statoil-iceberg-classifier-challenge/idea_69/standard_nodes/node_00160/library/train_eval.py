import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    CHECKPOINT_DIR,
    SUBMISSION_PATH,
    NUM_FOLDS,
    SUBMISSION_DIR,
)
from library.model import MCICNN
from library.data_loader import get_data_loaders, get_test_loader
from library.utils import set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B) -> (B, 1)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

    return running_loss / count


def train_fold(fold_idx, load_cached_data=True):
    """
    Trains the model for a specific fold using Early Stopping.
    Saves the best model checkpoint.
    """
    print(f"\n=== Training Fold {fold_idx} ===")

    # Ensure reproducibility
    set_seed()

    # Load data
    train_loader, val_loader = get_data_loaders(
        fold_idx, load_cached_data=load_cached_data
    )

    # Initialize model
    model = MCICNN().to(DEVICE)

    # Optimizer (AdamW with constant LR)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Loss Function (BCEWithLogitsLoss corresponds to Log Loss)
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.15f}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss:.15f}")
    return best_val_loss


def train_all_folds(load_cached_data=True):
    """
    Sequentially trains all folds defined in config.
    """
    fold_scores = []
    for fold_idx in range(NUM_FOLDS):
        score = train_fold(fold_idx, load_cached_data=load_cached_data)
        fold_scores.append(score)

    print("\n=== Cross-Validation Results ===")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score:.15f}")
    print(f"Average Log Loss: {np.mean(fold_scores):.15f}")


def predict_loader(model, loader, device):
    """
    Generates probability predictions for a given loader.
    Returns a dictionary mapping ID -> Probability.
    """
    model.eval()
    preds = {}

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for id_str, prob in zip(ids, probs):
                preds[id_str] = prob

    return preds


def generate_submission(load_cached_data=True):
    """
    Loads all fold models, generates predictions on the test set,
    averages them, and saves the submission file.
    """
    print("\n=== Generating Submission ===")

    # Load Test Data
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    # Dictionary to store sum of probabilities for averaging
    # We'll initialize it with the first fold
    accumulated_preds = {}

    # Iterate through all folds
    for fold_idx in range(NUM_FOLDS):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for fold {fold_idx} not found at {checkpoint_path}. Skipping."
            )
            continue

        print(f"Loading model for fold {fold_idx}...")
        model = MCICNN().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        fold_preds = predict_loader(model, test_loader, DEVICE)

        if not accumulated_preds:
            accumulated_preds = fold_preds
        else:
            for img_id, prob in fold_preds.items():
                accumulated_preds[img_id] += prob

    # Average predictions
    final_preds = []
    for img_id, prob_sum in accumulated_preds.items():
        avg_prob = prob_sum / NUM_FOLDS
        final_preds.append({"id": img_id, "is_iceberg": avg_prob})

    # Create DataFrame
    df_sub = pd.DataFrame(final_preds)

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(df_sub.head())
