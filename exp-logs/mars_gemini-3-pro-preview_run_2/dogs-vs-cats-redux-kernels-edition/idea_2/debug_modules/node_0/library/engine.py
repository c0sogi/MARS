import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_fold_dataloaders, get_test_dataloader
from library.model import EfficientNetClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (torch.device): The device to use.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (N, 1) for BCE

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion (Loss): The loss function.
        device (torch.device): The device to use.

    Returns:
        float: The average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return val_loss


def train_fold(fold_idx, epochs=Config.EPOCHS):
    """
    Trains a single fold of the K-Fold Cross Validation.

    Args:
        fold_idx (int): The index of the fold.
        epochs (int): Number of epochs to train.

    Returns:
        float: The best validation loss achieved.
    """
    seed_everything(Config.SEED + fold_idx)
    device = Config.DEVICE

    # Get DataLoaders
    train_loader, val_loader = get_fold_dataloaders(fold_idx)

    # Initialize Model
    model = EfficientNetClassifier()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}.pth")

    print(f"\nStarting training for Fold {fold_idx}...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), checkpoint_path)
            # print(f"  New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx} Best Val Loss: {best_val_loss:.10f}")
    return best_val_loss


def train_kfold(n_folds=Config.N_FOLDS, epochs=Config.EPOCHS):
    """
    Orchestrates the training of all K folds.

    Args:
        n_folds (int): Number of folds to train.
        epochs (int): Number of epochs per fold.
    """
    print(f"Training {n_folds} folds with {epochs} epochs each...")
    scores = []
    for fold in range(n_folds):
        score = train_fold(fold, epochs=epochs)
        scores.append(score)

    print("\nK-Fold Training Completed.")
    print(f"Average Validation Log Loss: {np.mean(scores):.10f}")


def generate_submission():
    """
    Generates predictions for the test set using an ensemble of all K-Fold models.
    Saves the submission file to Config.SUBMISSION_FILE.
    """
    print("\nGenerating submission...")
    device = Config.DEVICE
    test_loader = get_test_dataloader()

    # Dictionary to accumulate probabilities: id -> sum_of_probs
    id_to_prob_sum = {}
    id_to_count = {}

    # Iterate over all possible folds
    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold} not found. Skipping.")
            continue

        print(f"Predicting with Fold {fold} model...")

        model = EfficientNetClassifier()
        load_checkpoint(checkpoint_path, model, device=device)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                ids_np = ids.numpy().flatten()

                for img_id, prob in zip(ids_np, probs):
                    img_id = int(img_id)
                    if img_id not in id_to_prob_sum:
                        id_to_prob_sum[img_id] = 0.0
                        id_to_count[img_id] = 0
                    id_to_prob_sum[img_id] += prob
                    id_to_count[img_id] += 1

    # Average predictions
    results = []
    sorted_ids = sorted(id_to_prob_sum.keys())

    for img_id in sorted_ids:
        avg_prob = id_to_prob_sum[img_id] / id_to_count[img_id]
        results.append({"id": img_id, "label": avg_prob})

    submission_df = pd.DataFrame(results)

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
