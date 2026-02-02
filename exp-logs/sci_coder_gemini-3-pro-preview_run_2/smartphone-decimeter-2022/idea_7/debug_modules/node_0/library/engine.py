import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import meters_to_deg, calculate_competition_metric


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_x, batch_y, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        dataset_size += batch_x.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss (MAE) and the Competition Metric.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Tuple (val_loss, val_score).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Storage for metric calculation
    pred_records = []
    gt_records = []

    with torch.no_grad():
        for batch_x, batch_y, meta in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            dataset_size += batch_x.size(0)

            # Prepare data for metric calculation
            # Move to CPU numpy
            preds_m = outputs.cpu().numpy()
            targets_m = batch_y.cpu().numpy()

            # Metadata extraction
            trip_ids = meta["tripId"]  # list
            timestamps = meta["UnixTimeMillis"].numpy()
            wls_lats = meta["wls_lat"].numpy()
            wls_lons = meta["wls_lon"].numpy()

            # Convert predictions to degrees
            d_lat_deg_pred, d_lon_deg_pred = meters_to_deg(
                preds_m[:, 0], preds_m[:, 1], wls_lats
            )
            pred_lats = wls_lats + d_lat_deg_pred
            pred_lons = wls_lons + d_lon_deg_pred

            # Convert targets to degrees (to reconstruct GT)
            d_lat_deg_gt, d_lon_deg_gt = meters_to_deg(
                targets_m[:, 0], targets_m[:, 1], wls_lats
            )
            gt_lats = wls_lats + d_lat_deg_gt
            gt_lons = wls_lons + d_lon_deg_gt

            for i in range(len(trip_ids)):
                # Prediction Record
                pred_records.append(
                    {
                        "tripId": trip_ids[i],
                        "UnixTimeMillis": timestamps[i],
                        "LatitudeDegrees": pred_lats[i],
                        "LongitudeDegrees": pred_lons[i],
                    }
                )

                # Ground Truth Record
                gt_records.append(
                    {
                        "tripId": trip_ids[i],
                        "UnixTimeMillis": timestamps[i],
                        "LatitudeDegrees": gt_lats[i],
                        "LongitudeDegrees": gt_lons[i],
                    }
                )

    val_loss = running_loss / dataset_size

    # Calculate Competition Metric
    df_pred = pd.DataFrame(pred_records)
    df_gt = pd.DataFrame(gt_records)

    score = calculate_competition_metric(df_pred, df_gt)

    return val_loss, score


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Full training loop with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device.
        epochs: Max epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}, Val Score: {val_score:.10f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        device: Device.
        output_path: Path to save the CSV.
    """
    model.eval()
    results = []

    print("Generating predictions for submission...")

    with torch.no_grad():
        for batch_x, meta in test_loader:
            batch_x = batch_x.to(device)

            # Predict residuals (meters)
            preds_m = model(batch_x).cpu().numpy()

            # Metadata
            trip_ids = meta["tripId"]
            timestamps = meta["UnixTimeMillis"].numpy()
            wls_lats = meta["wls_lat"].numpy()
            wls_lons = meta["wls_lon"].numpy()

            # Convert metric residuals to degrees
            d_lat_deg, d_lon_deg = meters_to_deg(preds_m[:, 0], preds_m[:, 1], wls_lats)

            # Apply correction
            pred_lats = wls_lats + d_lat_deg
            pred_lons = wls_lons + d_lon_deg

            for i in range(len(trip_ids)):
                results.append(
                    {
                        "tripId": trip_ids[i],
                        "UnixTimeMillis": timestamps[i],
                        "LatitudeDegrees": pred_lats[i],
                        "LongitudeDegrees": pred_lons[i],
                    }
                )

    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
