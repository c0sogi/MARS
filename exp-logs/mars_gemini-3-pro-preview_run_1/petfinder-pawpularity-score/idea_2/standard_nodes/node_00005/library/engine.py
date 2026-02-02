import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_rmse


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        data_loader: The training data loader.
        device: The device to run training on.
        epoch: The current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    # Using BCEWithLogitsLoss as targets are scaled to [0, 1] in the dataset
    criterion = nn.BCEWithLogitsLoss()

    for batch_data in data_loader:
        # Unpack batch data
        images = batch_data["image"].to(device)
        features = batch_data["features"].to(device)
        targets = batch_data["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, features)

        # Flatten outputs and targets to ensure shape match
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    print(f"Epoch {epoch} Training Loss: {avg_loss}")

    return avg_loss


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: The validation data loader.
        device: The device to run evaluation on.

    Returns:
        tuple: (average_loss, rmse_score)
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_data in data_loader:
            images = batch_data["image"].to(device)
            features = batch_data["features"].to(device)
            targets = batch_data["target"].to(device)

            outputs = model(images, features)

            outputs = outputs.view(-1)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

            # Convert logits to probabilities [0, 1] then scale to [0, 100]
            preds = torch.sigmoid(outputs) * 100.0

            # Rescale targets from [0, 1] to [0, 100] for RMSE calculation
            targets_scaled = targets * 100.0

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets_scaled.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    rmse = calculate_rmse(all_targets, all_preds)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation RMSE: {rmse}")

    return avg_loss, rmse


def inference(model, data_loader, device, output_path=Config.submission_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        data_loader: The test data loader.
        device: The device to run inference on.
        output_path: Path to save the submission CSV.
    """
    model.eval()

    ids = []
    preds = []

    with torch.no_grad():
        for batch_data in data_loader:
            images = batch_data["image"].to(device)
            features = batch_data["features"].to(device)
            # Targets are not needed/available for inference

            outputs = model(images, features)
            outputs = outputs.view(-1)

            # Convert logits to probabilities and scale to [0, 100]
            batch_preds = torch.sigmoid(outputs) * 100.0

            ids.extend(batch_data["id"])
            preds.extend(batch_preds.cpu().numpy())

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Pawpularity": preds})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
