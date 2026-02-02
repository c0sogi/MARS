import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_device


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The neural network.
        loader: Training DataLoader (returns flattened views).
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (epoch_loss, epoch_auc)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        # Handle edge case with single class in batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using Dual-Anchor Consensus.

    Args:
        model: The neural network.
        loader: Validation DataLoader (returns grouped views).
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (val_loss, val_auc)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for v1, v2, labels, _ in loader:
            v1 = v1.to(device)
            v2 = v2.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward both views
            logits1 = model(v1)
            logits2 = model(v2)

            # Calculate loss as average of individual view losses
            # This aligns with the training objective
            loss1 = criterion(logits1, labels)
            loss2 = criterion(logits2, labels)
            loss = (loss1 + loss2) / 2.0

            running_loss += loss.item() * v1.size(0)

            # Consensus Probability for Metric
            prob1 = torch.sigmoid(logits1)
            prob2 = torch.sigmoid(logits2)
            avg_prob = (prob1 + prob2) / 2.0

            all_preds.append(avg_prob.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    return total_loss, auc


def predict_consensus(model, loader, device):
    """
    Generates predictions for the test set using Dual-Anchor Consensus + TTA.
    Saves the submission file to Config.SUBMISSION_PATH.

    Args:
        model: The neural network.
        loader: Test DataLoader.
        device: Torch device.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for v1, v2, _, ids in loader:
            v1 = v1.to(device)
            v2 = v2.to(device)

            # ------------------------------------------------------------------
            # Test-Time Augmentation (TTA)
            # ------------------------------------------------------------------
            # We generate 3 versions for EACH view: Original, Horizontal Flip, Vertical Flip
            # Total 6 predictions per subject to average.

            # View 1 Variants
            v1_orig = v1
            v1_hflip = torch.flip(v1, [3])  # Flip width
            v1_vflip = torch.flip(v1, [2])  # Flip height

            # View 2 Variants
            v2_orig = v2
            v2_hflip = torch.flip(v2, [3])
            v2_vflip = torch.flip(v2, [2])

            # Stack for efficient batch processing
            # Shape: (Batch * 6, Channels, H, W)
            batch_stack = torch.cat(
                [v1_orig, v1_hflip, v1_vflip, v2_orig, v2_hflip, v2_vflip], dim=0
            )

            # Inference
            logits = model(batch_stack)
            probs = torch.sigmoid(logits)  # (Batch * 6, 1)

            # Split back to average
            batch_size = v1.size(0)

            p_v1_o = probs[0:batch_size]
            p_v1_h = probs[batch_size : 2 * batch_size]
            p_v1_v = probs[2 * batch_size : 3 * batch_size]
            p_v2_o = probs[3 * batch_size : 4 * batch_size]
            p_v2_h = probs[4 * batch_size : 5 * batch_size]
            p_v2_v = probs[5 * batch_size : 6 * batch_size]

            # Average all 6 probabilities
            avg_probs = (p_v1_o + p_v1_h + p_v1_v + p_v2_o + p_v2_h + p_v2_v) / 6.0

            # Convert to numpy
            avg_probs_np = avg_probs.cpu().numpy().flatten()

            # Store results
            for i, subject_id in enumerate(ids):
                results.append({"BraTS21ID": subject_id, "MGMT_value": avg_probs_np[i]})

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    return df_sub


def train_model(
    model, train_loader, val_loader, optimizer, device, epochs=Config.EPOCHS
):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        model: The neural network.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: Torch device.
        epochs: Maximum number of epochs.

    Returns:
        model: The best model loaded from checkpoint.
    """
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Train Step
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validation Step
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Logging (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (AUC: {best_auc:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model before returning
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
