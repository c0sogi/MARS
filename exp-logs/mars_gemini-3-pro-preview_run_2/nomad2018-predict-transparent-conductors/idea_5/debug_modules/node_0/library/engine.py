import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import StandardScaler


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The PyTorch Geometric DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function (typically MSELoss).
        scaler: The StandardScaler to normalize targets.
        device: The device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data)

        # Scale targets to match model output distribution (zero mean, unit var)
        if data.y is not None:
            # Ensure targets are on the correct device and float32
            targets = data.y.to(device)
            scaled_targets = scaler.transform(targets)

            loss = criterion(out, scaled_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs

    return total_loss / num_graphs if num_graphs > 0 else 0.0


def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
):
    """
    Evaluates the model on a dataset.

    Args:
        model: The PyTorch model.
        loader: The PyTorch Geometric DataLoader.
        criterion: The loss function.
        scaler: The StandardScaler to inverse transform predictions.
        device: The device to run on.

    Returns:
        avg_loss (float): Average loss on scaled data.
        metrics (dict): Dictionary containing MAE and RMSE on original scale.
        predictions (dict): Dictionary mapping 'id' -> 'prediction' (numpy array).
    """
    model.eval()
    total_loss = 0.0
    num_graphs = 0

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Forward pass
            out = model(data)

            # Inverse transform to get predictions in original units (eV)
            preds_original = scaler.inverse_transform(out)

            # Store IDs and predictions
            all_ids.append(data.id.cpu().numpy())
            all_preds.append(preds_original.cpu().numpy())

            # If targets exist, calculate loss and store targets
            if data.y is not None:
                targets = data.y.to(device)
                scaled_targets = scaler.transform(targets)

                loss = criterion(out, scaled_targets)
                total_loss += loss.item() * data.num_graphs

                all_targets.append(targets.cpu().numpy())

            num_graphs += data.num_graphs

    # Aggregate results
    flat_ids = np.concatenate(all_ids)
    flat_preds = np.concatenate(all_preds, axis=0)

    metrics = {}
    avg_loss = total_loss / num_graphs if num_graphs > 0 else 0.0

    if all_targets:
        flat_targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics on original scale
        mae = np.mean(np.abs(flat_preds - flat_targets), axis=0)
        rmse = np.sqrt(np.mean((flat_preds - flat_targets) ** 2, axis=0))

        # Column-wise metrics
        metrics["mae_formation"] = mae[0]
        metrics["mae_bandgap"] = mae[1]
        metrics["rmse_formation"] = rmse[0]
        metrics["rmse_bandgap"] = rmse[1]

        # Overall metrics
        metrics["mae_avg"] = np.mean(mae)
        metrics["rmse_avg"] = np.mean(rmse)

    # Format predictions for easy access
    predictions = {"ids": flat_ids, "preds": flat_preds}

    return avg_loss, metrics, predictions
