import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.data_processing import get_dataloaders
from library.models import SPIRVNet, FocalLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Unpack batch
        if len(batch) == 3:
            kin, vis, targets = batch
            targets = targets.to(device)
        else:
            # Should not happen in training, but safe fallback
            kin, vis = batch
            targets = None

        kin = kin.to(device)
        vis = vis.to(device)

        batch_size = kin.size(0)
        dataset_size += batch_size

        # Forward
        optimizer.zero_grad()
        logits = model(kin, vis)

        # Loss
        loss = criterion(logits, targets.unsqueeze(1))

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, all probabilities, and all true labels.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            kin, vis, targets = batch
            kin = kin.to(device)
            vis = vis.to(device)
            targets = targets.to(device)

            batch_size = kin.size(0)
            dataset_size += batch_size

            logits = model(kin, vis)
            loss = criterion(logits, targets.unsqueeze(1))

            running_loss += loss.item() * batch_size

            # Store probabilities (sigmoid) and targets for MCC calculation
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
        all_targets = np.concatenate(all_targets).flatten()
    else:
        all_probs = np.array([])
        all_targets = np.array([])

    return avg_loss, all_probs, all_targets


def find_optimal_threshold(y_true, y_probs):
    """
    Finds the threshold that maximizes MCC on the provided data.
    """
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_mcc = -1.0
    best_thresh = 0.5

    # Ensure integer types for y_true
    y_true = y_true.astype(int)

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        score = matthews_corrcoef(y_true, preds)

        if score > best_mcc:
            best_mcc = score
            best_thresh = th

    return best_thresh, best_mcc


def predict_test(model, loader, threshold, device):
    """
    Generates binary predictions for the test set using a specific threshold.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns (kin, vis)
            kin, vis = batch
            kin = kin.to(device)
            vis = vis.to(device)

            logits = model(kin, vis)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
    else:
        all_probs = np.array([])

    # Apply threshold
    predictions = (all_probs >= threshold).astype(int)
    return predictions


def run_training(epochs=Config.EPOCHS, load_cached_data=True):
    """
    Main orchestration function.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    # Determine input dimensions from a batch
    dummy_kin, dummy_vis, _ = next(iter(train_loader))
    input_dim_kin = dummy_kin.shape[1]
    input_dim_vis = dummy_vis.shape[1]

    print(
        f"Initializing SPIRVNet with Kinematic Dim: {input_dim_kin}, Visual Dim: {input_dim_vis}"
    )

    model = SPIRVNet(input_dim_kin, input_dim_vis).to(device)

    # 3. Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 4. Training Loop
    best_mcc = -1.0
    best_threshold_val = 0.5
    patience_counter = 0

    threshold_save_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Optimize Threshold
        curr_thresh, curr_mcc = find_optimal_threshold(val_targets, val_probs)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCC: {curr_mcc} | Best Thresh: {curr_thresh}"
        )

        # Early Stopping & Checkpointing
        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold_val = curr_thresh
            patience_counter = 0

            print(f"New best MCC: {best_mcc}. Saving model...")
            save_checkpoint(model, optimizer, epoch, best_mcc, Config.MODEL_SAVE_PATH)

            # Save threshold
            np.save(threshold_save_path, np.array([best_threshold_val]))
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference & Submission
    print("\nStarting Inference...")

    # Load best model
    checkpoint = load_checkpoint(model, Config.MODEL_SAVE_PATH, device=Config.DEVICE)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with MCC {checkpoint['score']}"
    )

    # Load best threshold
    if os.path.exists(threshold_save_path):
        best_threshold_val = np.load(threshold_save_path)[0]
        print(f"Loaded optimized threshold: {best_threshold_val}")
    else:
        print(
            f"Warning: Threshold file not found. Using last best: {best_threshold_val}"
        )

    # Generate Predictions
    predictions = predict_test(model, test_loader, best_threshold_val, device)

    # Create Submission File
    # We read metadata/test.csv to get the contact_ids in the correct order.
    # The test_loader iterates over metadata/test.csv sequentially.
    df_test_meta = pd.read_csv(Config.METADATA_TEST)

    if len(df_test_meta) != len(predictions):
        raise ValueError(
            f"Mismatch in prediction length: Metadata {len(df_test_meta)} vs Preds {len(predictions)}"
        )

    df_test_meta["contact"] = predictions

    # Select required columns
    submission_df = df_test_meta[["contact_id", "contact"]]

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
