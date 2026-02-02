import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
import library.config as config


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for features, targets, is_ground in loader:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1)
        is_ground = is_ground.to(device)

        optimizer.zero_grad()

        logits = model(features, is_ground)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        num_samples += features.size(0)

    return total_loss / num_samples if num_samples > 0 else 0.0


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, mcc, probabilities, targets)
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for features, targets, is_ground in loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            is_ground = is_ground.to(device)

            logits = model(features, is_ground)
            loss = criterion(logits, targets)

            total_loss += loss.item() * features.size(0)
            num_samples += features.size(0)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        # Calculate MCC with default threshold of 0.5 for monitoring
        preds_binary = (all_probs > 0.5).astype(int)
        mcc = matthews_corrcoef(all_targets, preds_binary)
    else:
        all_probs = np.array([])
        all_targets = np.array([])
        mcc = 0.0

    return avg_loss, mcc, all_probs, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=config.EPOCHS,
    lr=config.LEARNING_RATE,
    patience=config.EARLY_STOPPING_PATIENCE,
    pos_weight=config.POS_WEIGHT,
    save_path=config.MODEL_SAVE_PATH,
):
    """
    Main training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        pos_weight (float): Positive class weight for BCE loss.
        save_path (str): Path to save the best model.

    Returns:
        nn.Module: The trained model (loaded with best weights).
    """
    # Set seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Weighted BCE Loss to handle class imbalance
    # pos_weight must be a tensor on the same device
    weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_mcc = -1.0
    patience_counter = 0

    print(f"Starting training on {device}")
    print(
        f"Hyperparameters: Epochs={epochs}, LR={lr}, PosWeight={pos_weight}, Patience={patience}"
    )

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcc, _, _ = evaluate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCC: {val_mcc}"
        )

        # Early Stopping & Checkpointing
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with MCC: {best_mcc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCC: {best_mcc}")

    # Load best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def optimize_threshold(model, val_loader):
    """
    Finds the optimal decision threshold on the validation set to maximize MCC.

    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.

    Returns:
        float: The optimal threshold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # We use a simple criterion placeholder since we only need probs/targets
    # pos_weight doesn't matter for evaluation metric calculation here
    criterion = nn.BCEWithLogitsLoss()

    _, _, all_probs, all_targets = evaluate(model, val_loader, criterion, device)

    # Grid search for threshold
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_thresh = 0.5
    best_mcc = -1.0

    for t in thresholds:
        preds = (all_probs > t).astype(int)
        mcc = matthews_corrcoef(all_targets, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    print(f"Threshold Optimization: Best Threshold = {best_thresh} (MCC: {best_mcc})")
    return best_thresh


def generate_submission(
    model, test_loader, threshold, output_path=config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        threshold (float): Decision threshold.
        output_path (str): Path to save the submission CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for features, _, is_ground in test_loader:
            features = features.to(device)
            is_ground = is_ground.to(device)

            logits = model(features, is_ground)
            probs = torch.sigmoid(logits)

            # Apply threshold
            preds = (probs > threshold).int().cpu().numpy()
            all_preds.append(preds)

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds).flatten()
    else:
        all_preds = np.array([])

    # Load sample submission to get IDs and structure
    sub_df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Sanity check on lengths
    if len(all_preds) != len(sub_df):
        print(
            f"Warning: Prediction length ({len(all_preds)}) does not match submission length ({len(sub_df)})."
        )
        # In a real scenario, we would handle alignment carefully.
        # Here we assume the loader preserves order and length matches metadata generation.

    sub_df["contact"] = all_preds

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
