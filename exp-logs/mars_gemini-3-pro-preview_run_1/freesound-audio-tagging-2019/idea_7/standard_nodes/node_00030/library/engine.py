import os
import numpy as np
import torch
import torch.nn as nn
from library.config import CFG
from library.utils import mixup_data, mixup_criterion, calculate_lwlrap


def train_fn(model, data_loader, optimizer, device):
    """
    Performs one epoch of training with Mixup augmentation.

    Args:
        model: The PyTorch model.
        data_loader: Training DataLoader.
        optimizer: The optimizer.
        device: Computation device (cuda/cpu).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define loss function (Mixup handles the combination)
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)

        # Apply Mixup (CFG.mixup_prob is 1.0)
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, CFG.mixup_alpha, device
        )

        optimizer.zero_grad()

        outputs = model(inputs)

        # Calculate Mixup loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size for accurate mean)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        data_loader: Validation DataLoader.
        device: Computation device.

    Returns:
        tuple: (Average Loss, LWLRAP Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    predictions = np.concatenate(preds_list, axis=0)
    ground_truth = np.concatenate(targets_list, axis=0)

    # Calculate LWLRAP
    score = calculate_lwlrap(ground_truth, predictions)

    return epoch_loss, score


def inference_fn(model, data_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        data_loader: Test DataLoader.
        device: Computation device.

    Returns:
        np.ndarray: Predicted probabilities (N_samples, N_classes).
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            # Apply Sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)
    return predictions


def save_submission(predictions, test_df, save_path="./submission/submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Predicted probabilities.
        test_df (pd.DataFrame): Test DataFrame containing 'fname'.
        save_path (str): Path to save the submission file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a copy to avoid modifying the original dataframe
    submit_df = test_df.copy()

    # Assign predictions to the class columns
    # CFG.target_columns defines the correct order of classes
    submit_df[CFG.target_columns] = predictions

    # Select only the required columns: fname + class columns
    cols = ["fname"] + CFG.target_columns
    submit_df = submit_df[cols]

    # Save to CSV
    submit_df.to_csv(save_path, index=False)
