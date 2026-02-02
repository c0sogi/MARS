import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from library.config import (
    DEVICE,
    OUTPUT_DIR,
    SUBMISSION_PATH,
    LR,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE,
    SEED,
    NUM_TAGS,
)
from library.model import WideAndDeepModel, FocalLoss
from library.dataset import get_dataloaders


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move batch data to device
        deep_seq = batch["deep_seq"].to(device)
        wide_indices = batch["wide_indices"].to(device)
        wide_values = batch["wide_values"].to(device)
        wide_offsets = batch["wide_offsets"].to(device)
        targets = batch["target"].to(device)

        batch_size = targets.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            probs = model(deep_seq, wide_indices, wide_values, wide_offsets)
            loss = criterion(probs, targets)

        # Backward pass with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a dataset.
    Returns:
        avg_loss: Scalar float
        all_probs: Numpy array of shape (n_samples, num_tags)
        all_targets: Numpy array of shape (n_samples, num_tags)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            deep_seq = batch["deep_seq"].to(device)
            wide_indices = batch["wide_indices"].to(device)
            wide_values = batch["wide_values"].to(device)
            wide_offsets = batch["wide_offsets"].to(device)
            targets = batch["target"].to(device)

            batch_size = targets.size(0)
            dataset_size += batch_size

            # Forward pass (Mixed Precision is optional for eval but good for consistency)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                probs = model(deep_seq, wide_indices, wide_values, wide_offsets)
                loss = criterion(probs, targets)

            running_loss += loss.item() * batch_size

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if all_probs:
        all_probs = np.vstack(all_probs)
        all_targets = np.vstack(all_targets)
    else:
        all_probs = np.array([])
        all_targets = np.array([])

    return avg_loss, all_probs, all_targets


def find_best_threshold(probs, targets):
    """
    Finds the optimal probability threshold that maximizes the sample-averaged F1 score.
    """
    best_thresh = 0.5
    best_score = 0.0

    # Search range from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.95, 0.05)

    for t in thresholds:
        preds = (probs > t).astype(int)
        # Calculate Mean F1-Score (samples average)
        score = f1_score(targets, preds, average="samples", zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def run_training(load_cached_data=True):
    """
    Main pipeline: Data Loading -> Training -> Validation -> Inference -> Submission.
    """
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids, preprocessor = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # ---------------------------------------------------------
    # 2. Initialize Model & Optimizer
    # ---------------------------------------------------------
    print("Initializing model...")
    model = WideAndDeepModel().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, criterion, DEVICE
        )

        # Monitor F1 with default threshold
        val_preds_default = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(
            val_targets, val_preds_default, average="samples", zero_division=0
        )

        print(
            f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f} - Val F1 (0.5): {val_f1:.8f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved!")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # ---------------------------------------------------------
    # 4. Threshold Optimization
    # ---------------------------------------------------------
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    print("Optimizing threshold on validation set...")
    # Re-run evaluation to get probs of the best model
    _, val_probs, val_targets = evaluate(model, val_loader, criterion, DEVICE)
    best_thresh, best_f1 = find_best_threshold(val_probs, val_targets)
    print(f"Best Threshold: {best_thresh:.4f} - Best Val F1: {best_f1:.8f}")

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    print("Generating predictions for test set...")
    # Note: Test targets are dummy zeros, ignore them
    _, test_probs, _ = evaluate(model, test_loader, criterion, DEVICE)

    # Apply optimized threshold
    test_preds_binary = (test_probs > best_thresh).astype(int)

    print("Converting predictions to tags...")
    # Inverse transform using the MultiLabelBinarizer from preprocessor
    test_tags = preprocessor.mlb.inverse_transform(test_preds_binary)

    # Join tags with space
    test_tags_str = [" ".join(tags) for tags in test_tags]

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Tags": test_tags_str})

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")
