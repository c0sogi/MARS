import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import load_df


def calculate_pos_weights(load_cached_data=True, device="cpu"):
    """
    Calculates positive weights for BCEWithLogitsLoss to handle class imbalance.
    Uses caching mechanism to store weights in .npy format.

    Args:
        load_cached_data (bool): Whether to try loading from cache.
        device (str): Device to move the weights tensor to.

    Returns:
        torch.Tensor: Weights for each class.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "pos_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(device)
        except Exception:
            # If loading fails, proceed to compute
            pass

    # 2. Compute from scratch
    # Load training metadata
    df = load_df("train")

    # Get target labels
    targets = df[Config.TARGET_COLS].values

    # Calculate counts
    pos_counts = np.sum(targets, axis=0)
    total_counts = len(df)
    neg_counts = total_counts - pos_counts

    # Calculate weights: number of negatives / number of positives
    # Add a small epsilon to pos_counts to avoid division by zero
    weights = neg_counts / (pos_counts + 1e-6)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, dataloader, optimizer, device, pos_weights):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer.
        device: Device to run training on.
        pos_weights: Class weights for loss function.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: Device to run evaluation on.

    Returns:
        tuple: (average_auc, dict_of_individual_aucs)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    if not all_preds:
        return 0.0, {}

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    auc_scores = {}
    valid_aucs = []

    for i, col_name in enumerate(Config.TARGET_COLS):
        try:
            # Only calculate AUC if there are at least two classes (0 and 1)
            if len(np.unique(all_targets[:, i])) > 1:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                auc_scores[col_name] = score
                valid_aucs.append(score)
            else:
                # Fallback for single-class targets in validation batch
                auc_scores[col_name] = 0.5
                valid_aucs.append(0.5)
        except ValueError:
            auc_scores[col_name] = 0.5
            valid_aucs.append(0.5)

    avg_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    return avg_auc, auc_scores


def generate_submission(model, dataloader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for test data.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    if not all_preds:
        return

    all_preds = np.concatenate(all_preds, axis=0)

    # Load test metadata to get StudyInstanceUIDs
    # We assume the dataloader preserves the order of the metadata
    df_test = load_df("test")

    # Create submission DataFrame
    submission_df = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "StudyInstanceUID", df_test["StudyInstanceUID"])

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
