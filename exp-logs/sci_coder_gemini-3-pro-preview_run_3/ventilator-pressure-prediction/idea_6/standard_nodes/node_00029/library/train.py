import torch
import torch.optim as optim
import os
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import (
    PCANet,
    MaskedMAELoss,
    train_epoch,
    validate,
    predict_and_submit,
)
from library.utils import clear_cache


def train_model(
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    load_cached_data: bool = True,
):
    """
    Orchestrates the training pipeline for the PCA-Net model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate for the optimizer.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training run on device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Handle cache clearing if forcing a fresh start
    if not load_cached_data:
        clear_cache(Config.WORKING_DIR)

    # 2. Data Loading
    # The get_dataloaders function handles the complexity of feature engineering and caching
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    model = PCANet(Config).to(device)

    # 4. Optimization Setup
    criterion = MaskedMAELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    early_stop_counter = 0

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Execute one training epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Execute validation
        val_loss = validate(model, val_loader, criterion, device)

        # Update Learning Rate
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(f"Epoch {epoch+1} | Train MAE: {train_loss} | Val MAE: {val_loss}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 6. Inference and Submission
    print("Training complete. Loading best model for inference...")

    # Load the best saved state
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    print("Generating submission file...")
    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
