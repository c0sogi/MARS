import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library import config
from library import dataset
from library import model
from library import utils

# Set seed for reproducibility
config.set_seed()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: predict noise residual
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss (MSE) and RMSE.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    # RMSE of residuals is equivalent to RMSE of reconstructed images
    # because (Input - Pred_Noise) - (Input - True_Noise) = True_Noise - Pred_Noise
    epoch_rmse = np.sqrt(epoch_loss)

    return epoch_loss, epoch_rmse


def train_model(load_cached_data=True):
    """
    Main function to train the RDN model.
    """
    device = config.DEVICE
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_dataset, val_dataset = dataset.prepare_datasets(
        load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # 2. Initialize Model
    rdn = model.RDN(
        channel=config.IMG_CHANNELS,
        growth_rate=config.RDN_GROWTH_RATE,
        num_features=config.RDN_NUM_FEATURES,
        num_blocks=config.RDN_NUM_BLOCKS,
        num_layers=config.RDN_LAYERS_PER_BLOCK,
        kernel_size=config.RDN_KERNEL_SIZE,
    ).to(device)

    # 3. Setup Optimization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        rdn.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
    )

    # 4. Training Loop
    best_rmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(rdn, train_loader, criterion, optimizer, device)
        val_loss, val_rmse = validate(rdn, val_loader, criterion, device)

        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val RMSE: {val_rmse}"
        )

        # Checkpoint
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(rdn.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with RMSE: {best_rmse}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val RMSE: {best_rmse}")
    return rdn


def predict_and_submit():
    """
    Loads the best model, performs inference on the test set, and generates the submission file.
    """
    device = config.DEVICE

    # Load Test Metadata
    if not os.path.exists(config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(config.TEST_METADATA_PATH)
    test_ids = df_test["image_id"].tolist()
    input_paths = df_test["input_path"].tolist()

    # Initialize Model
    rdn = model.RDN(
        channel=config.IMG_CHANNELS,
        growth_rate=config.RDN_GROWTH_RATE,
        num_features=config.RDN_NUM_FEATURES,
        num_blocks=config.RDN_NUM_BLOCKS,
        num_layers=config.RDN_LAYERS_PER_BLOCK,
        kernel_size=config.RDN_KERNEL_SIZE,
    ).to(device)

    # Load Weights
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model weights not found at {config.MODEL_SAVE_PATH}")

    rdn.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    rdn.eval()

    predictions = []

    print(f"Starting inference on {len(test_ids)} test images...")

    with torch.no_grad():
        for rel_path in input_paths:
            full_path = os.path.join(config.INPUT_DIR, rel_path)

            # Load full image (normalized [0, 1])
            img_noisy = utils.load_grayscale_image(full_path)

            # Prepare input tensor: (1, 1, H, W)
            input_tensor = (
                torch.from_numpy(img_noisy).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict noise residual
            noise_pred = rdn(input_tensor)

            # Reconstruct clean image: Clean = Noisy - Noise
            clean_pred = input_tensor - noise_pred

            # Remove batch and channel dims -> (H, W)
            clean_pred_np = clean_pred.squeeze().cpu().numpy()

            # Clip to valid range
            clean_pred_np = np.clip(clean_pred_np, 0.0, 1.0)

            predictions.append(clean_pred_np)

    # Generate Submission
    utils.format_submission(test_ids, predictions, config.SUBMISSION_PATH)
