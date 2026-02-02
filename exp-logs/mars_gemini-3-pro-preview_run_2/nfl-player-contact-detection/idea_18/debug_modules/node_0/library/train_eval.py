import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, calc_mcc, FocalLoss
from library.models import RVCNet
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for (k_input, v_input), targets in loader:
        batch_size = targets.size(0)
        k_input = k_input.to(device)
        v_input = v_input.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        logits = model(k_input, v_input)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns flattened logits and ground truth labels.
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for (k_input, v_input), targets in loader:
            k_input = k_input.to(device)
            v_input = v_input.to(device)

            logits = model(k_input, v_input)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    all_logits = np.concatenate(all_logits).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    return all_logits, all_targets


def optimize_threshold(y_true, y_logits):
    """
    Finds the decision threshold that maximizes MCC on the validation set.
    """
    # Convert logits to probabilities
    y_probs = 1 / (1 + np.exp(-y_logits))

    best_threshold = 0.5
    best_mcc = -1.0

    thresholds = np.arange(
        Config.THRESHOLD_SEARCH_START,
        Config.THRESHOLD_SEARCH_END,
        Config.THRESHOLD_SEARCH_STEP,
    )

    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)
        mcc = calc_mcc(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    return best_threshold, best_mcc


def train_model(load_cached_data=True):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Initialize Model
    model = RVCNet().to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss()

    # Training State
    best_val_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_logits, val_targets = validate(model, val_loader, device)

        # Optimize Threshold
        curr_threshold, curr_mcc = optimize_threshold(val_targets, val_logits)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCC: {curr_mcc:.10f} | "
            f"Best Thresh: {curr_threshold:.4f}"
        )

        # Early Stopping & Checkpointing
        if curr_mcc > best_val_mcc:
            best_val_mcc = curr_mcc
            best_threshold = curr_threshold
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)

            # Save best threshold for inference
            np.save(
                os.path.join(Config.WORKING_DIR, "best_threshold.npy"),
                np.array([best_threshold]),
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(
        f"Training complete. Best Val MCC: {best_val_mcc:.10f} at Threshold: {best_threshold:.4f}"
    )
    return best_threshold


def generate_submission(threshold=None):
    """
    Generates submission file using the best trained model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Determine Threshold
    if threshold is None:
        thresh_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")
        if os.path.exists(thresh_path):
            threshold = float(np.load(thresh_path)[0])
        else:
            print("Warning: No threshold found. Defaulting to 0.5")
            threshold = 0.5

    print(f"Generating submission with threshold: {threshold:.4f}")

    # Load Test Data
    # Note: We use get_dataloaders to ensure consistency, but only need test_loader
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    model = RVCNet().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Inference
    all_logits = []
    with torch.no_grad():
        for (k_input, v_input), _ in test_loader:
            k_input = k_input.to(device)
            v_input = v_input.to(device)

            logits = model(k_input, v_input)
            all_logits.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits).flatten()

    # Apply Threshold
    probs = 1 / (1 + np.exp(-all_logits))
    predictions = (probs > threshold).astype(int)

    # Retrieve Test IDs
    # These are saved by data_processing.py in the cache directory
    ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    if os.path.exists(ids_path):
        test_ids = np.load(ids_path, allow_pickle=True)
    else:
        raise FileNotFoundError(
            f"Test IDs not found at {ids_path}. Run data processing first."
        )

    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions."
        )

    # Save Submission
    df_sub = pd.DataFrame({"contact_id": test_ids, "contact": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
