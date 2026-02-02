import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import RelativeTrajectoryCNN, GNSSWindowDataset, prepare_data
from library.utils import enu_to_ecef, ecef_to_lla


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        # Accumulate loss weighted by batch size
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_epoch(dataloader, model, loss_fn, device):
    """
    Evaluates the model on the validation set.

    Args:
        dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        device (str): Device to run evaluation on.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_training(debug_size=None, epochs=Config.NUM_EPOCHS, load_cached_data=True):
    """
    Orchestrates the full training process including data loading, model initialization,
    training loop, validation, scheduler stepping, and early stopping.

    Args:
        debug_size (int, optional): Limit dataset size for debugging. Defaults to None.
        epochs (int, optional): Maximum number of epochs. Defaults to Config.NUM_EPOCHS.
        load_cached_data (bool, optional): Whether to load pre-processed data from cache. Defaults to True.
    """
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = Config.DEVICE
    print(f"Running training on device: {device}")

    # 1. Load Data
    print("Loading training data...")
    X_train, y_train, _ = prepare_data(
        Config.TRAIN_METADATA_PATH,
        mode="train",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    print("Loading validation data...")
    X_val, y_val, _ = prepare_data(
        Config.VAL_METADATA_PATH,
        mode="val",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    # 2. Create DataLoaders
    train_dataset = GNSSWindowDataset(X_train, y_train)
    val_dataset = GNSSWindowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Initialize Model
    model = RelativeTrajectoryCNN(
        input_channels=Config.NUM_INPUT_FEATURES,
        hidden_channels=Config.CNN_HIDDEN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        fc_dim=Config.FC_HIDDEN_DIM,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    # 4. Setup Loss, Optimizer, Scheduler
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = validate_epoch(val_loader, model, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train MAE: {train_loss}")
        print(f"Val MAE: {val_loss}")

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Early Stopping and Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Val MAE: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Training complete. Best Validation MAE: {best_val_loss}")


def generate_submission(debug_size=None, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug_size (int, optional): Limit dataset size for debugging. Defaults to None.
        load_cached_data (bool, optional): Whether to load pre-processed data from cache. Defaults to True.
    """
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = Config.DEVICE
    print(f"Generating submission on device: {device}")

    # 1. Load Test Data
    print("Loading test data...")
    X_test, _, df_test_meta = prepare_data(
        Config.TEST_METADATA_PATH,
        mode="test",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    test_dataset = GNSSWindowDataset(X_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 2. Load Model
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train the model first."
        )

    model = RelativeTrajectoryCNN(
        input_channels=Config.NUM_INPUT_FEATURES,
        hidden_channels=Config.CNN_HIDDEN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        fc_dim=Config.FC_HIDDEN_DIM,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 3. Inference
    predictions = []
    print("Running inference...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    # Stack predictions: Shape (N, 2) -> [Delta East, Delta North]
    predictions = np.vstack(predictions)

    # 4. Reconstruct Absolute Coordinates
    print("Reconstructing absolute coordinates...")

    # Extract baseline WLS coordinates from metadata
    wls_lat = df_test_meta["WlsLat"].values
    wls_lon = df_test_meta["WlsLon"].values
    wls_alt = df_test_meta["WlsAlt"].values

    pred_e = predictions[:, 0]
    pred_n = predictions[:, 1]
    # We assume 0 vertical offset as we only predict horizontal residuals
    pred_u = np.zeros_like(pred_e)

    # Convert predicted ENU offsets back to ECEF relative to the WLS baseline
    pred_x, pred_y, pred_z = enu_to_ecef(
        pred_e, pred_n, pred_u, wls_lat, wls_lon, wls_alt
    )

    # Convert ECEF to Geodetic (Lat, Lon)
    pred_lat, pred_lon, _ = ecef_to_lla(pred_x, pred_y, pred_z)

    # 5. Save Submission
    submission = pd.DataFrame(
        {
            "tripId": df_test_meta["tripId"],
            "UnixTimeMillis": df_test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
