import os
import torch
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SLHDAN, train_one_epoch, validate, predict_and_submit


def run_training(debug=False, epochs=None):
    """
    Main execution function for the SLH-DAN training pipeline.

    Args:
        debug (bool): If True, uses a small subset of data for debugging purposes.
        epochs (int, optional): If provided, overrides the number of epochs in Config.
    """
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Configuration Overrides
    if epochs is not None:
        Config.EPOCHS = epochs
        Config.T_MAX = epochs  # Ensure scheduler aligns with new epoch count

    print(f"Starting training run on {device}. Epochs: {Config.EPOCHS}, Debug: {debug}")

    # 3. Data Loading
    # get_dataloaders handles metadata reading and dataset creation (including caching)
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 4. Model Initialization
    model = SLHDAN().to(device)

    # 5. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 6. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validation Step
        val_loss, val_metric = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Metric: {val_metric}")

        # Checkpointing (Save Best Model)
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! Metric: {best_metric}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")

    # 7. Inference and Submission
    # Loads the best model from disk, generates predictions, and saves submission.csv
    print("Generating submission...")
    predict_and_submit(test_loader)
