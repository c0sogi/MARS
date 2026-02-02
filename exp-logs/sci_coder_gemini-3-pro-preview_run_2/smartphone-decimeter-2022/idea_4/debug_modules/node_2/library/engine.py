import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library import config
from library import utils


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for window, context, target in loader:
        window = window.to(device)
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(window, context)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for window, context, target in loader:
            window = window.to(device)
            context = context.to(device)
            target = target.to(device)

            output = model(window, context)
            loss = criterion(output, target)

            total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss


def train_model(model, train_loader, val_loader, device, config_params):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    print("Starting training...")

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config_params["scheduler_factor"],
        patience=config_params["scheduler_patience"],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = config.CACHE_FILES["model"]

    # Ensure directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(config_params["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val Loss = {val_loss}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config_params["patience"]:
                print(f"  Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # Load the best model weights
    model.load_state_dict(torch.load(best_model_path))
    return model


def generate_submission(model, test_loader, test_df, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()
    predictions = []

    # Run inference
    with torch.no_grad():
        for window, context, _ in test_loader:
            window = window.to(device)
            context = context.to(device)

            output = model(window, context)
            predictions.append(output.cpu().numpy())

    # Concatenate all batch predictions: Shape (N, 2) -> (DeltaEast, DeltaNorth)
    predictions = np.concatenate(predictions, axis=0)

    # The Dataset iterates over groups sorted by tripId, and within trip sorted by time.
    # We must sort the test_df identically to align predictions with metadata.
    # Note: groupby in pandas sorts keys by default.
    sorted_df = test_df.sort_values(by=["tripId", "UnixTimeMillis"]).reset_index(
        drop=True
    )

    # Extract Baseline WLS coordinates
    wls_lat = sorted_df["WlsLat"].values
    wls_lon = sorted_df["WlsLon"].values
    wls_alt = sorted_df["WlsAlt"].values

    # Extract Predicted Residuals
    delta_east = predictions[:, 0]
    delta_north = predictions[:, 1]
    # We assume DeltaUp is 0 for the 2D correction, or we could model it.
    # Here we only predict 2D, so up=0.
    delta_up = np.zeros_like(delta_east)

    # Reconstruct Final Coordinates
    pred_lat, pred_lon, _ = utils.enu_to_lla(
        delta_east, delta_north, delta_up, wls_lat, wls_lon, wls_alt
    )

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": sorted_df["tripId"],
            "UnixTimeMillis": sorted_df["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Save to file
    utils.save_submission(submission, config.SUBMISSION_PATH)
