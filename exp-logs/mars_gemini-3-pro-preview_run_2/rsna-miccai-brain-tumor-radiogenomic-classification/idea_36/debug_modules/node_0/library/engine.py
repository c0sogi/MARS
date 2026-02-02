import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
    WORKING_DIR,
)
from library.model_factory import get_model
from library.data_factory import get_dataloaders


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass (model returns logits)
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count if count > 0 else 0.0


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set and computes ROC AUC.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.numpy()

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_targets.extend(targets)
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Handle edge case where only one class is present in batch/set
    if len(np.unique(all_targets)) < 2:
        return 0.5

    auc = roc_auc_score(all_targets, all_probs)
    return auc


def predict_with_tta(model, loader, device, submission_path):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA Strategy: Average predictions of Original, Horizontal Flip, and Vertical Flip.
    """
    model.eval()
    results = []

    print("Starting Inference with TTA...")

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)
            # inputs shape: (B, C, H, W)

            # 1. Original
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is Width)
            inputs_h = torch.flip(inputs, dims=[3])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is Height)
            inputs_v = torch.flip(inputs, dims=[2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            avg_probs = avg_probs.cpu().numpy().flatten()

            ids_np = ids.numpy()

            for i in range(len(ids_np)):
                results.append({"BraTS21ID": ids_np[i], "MGMT_value": avg_probs[i]})

    # Save Submission
    df_sub = pd.DataFrame(results)
    # Ensure BraTS21ID is sorted or formatted if necessary, though competition usually matches IDs
    # Sorting by ID for consistency
    df_sub = df_sub.sort_values("BraTS21ID")

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training(load_cached_data=True):
    """
    Main execution function:
    1. Sets seeds.
    2. Loads data.
    3. Trains model with Early Stopping.
    4. Saves best model.
    5. Runs inference.
    """
    set_seed(SEED)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = get_model()

    # 3. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_auc = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved! AUC: {best_auc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print("Warning: No model file found. Using current model state.")

    predict_with_tta(model, test_loader, DEVICE, SUBMISSION_PATH)
