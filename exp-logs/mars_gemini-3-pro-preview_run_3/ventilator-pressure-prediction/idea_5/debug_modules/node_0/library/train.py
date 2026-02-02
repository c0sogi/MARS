import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import CAPNet, train_epoch, validate, generate_submission


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    debug=Config.DEBUG,
    clean_start=Config.CLEAN_START,
    load_cache=Config.LOAD_CACHE,
):
    """
    Executes the training pipeline for the CAP-Net model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate.
        debug (bool): If True, runs on a smaller subset of data.
        clean_start (bool): If True, clears the cache before starting.
        load_cache (bool): If True, attempts to load processed data from cache.
    """

    # 1. Update Configuration based on arguments
    Config.EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = learning_rate
    Config.DEBUG = debug
    Config.CLEAN_START = clean_start
    Config.LOAD_CACHE = load_cache

    # 2. Setup Environment
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Initializing training on device: {device}")

    # 3. Cache Management
    # If clean_start is True, remove existing cache files to force re-processing
    if Config.CLEAN_START:
        print(f"Cleaning cache directory: {Config.WORKING_DIR}")
        if os.path.exists(Config.WORKING_DIR):
            for f in os.listdir(Config.WORKING_DIR):
                if f.endswith(".npy") or f.endswith(".pth"):
                    try:
                        os.remove(os.path.join(Config.WORKING_DIR, f))
                    except OSError as e:
                        print(f"Warning: Could not delete {f}: {e}")

    # 4. Data Loading
    # get_dataloaders handles the logic of computing vs loading based on Config.LOAD_CACHE
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=Config.LOAD_CACHE
    )

    # 5. Model Initialization
    model = CAPNet().to(device)

    # 6. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    # 7. Training Loop
    best_mae = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Training Step
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validation Step
        val_mae = validate(model, val_loader, device)

        # Scheduler Update
        scheduler.step(val_mae)

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_mae < best_mae:
            best_mae = val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! MAE: {best_mae:.6f}")
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 8. Final Inference and Submission
    print("Training complete. Loading best model for inference...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    print(f"Generating submission to {Config.SUBMISSION_PATH}...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    print("Pipeline finished successfully.")
