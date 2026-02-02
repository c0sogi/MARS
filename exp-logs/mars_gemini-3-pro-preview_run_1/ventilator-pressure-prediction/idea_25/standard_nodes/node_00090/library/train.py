import torch
import os
from library.config import Config
from library.utils import set_seed, get_device, save_checkpoint
from library.data import get_dataloaders
from library.model import VentilatorModel, train_epoch, validate, generate_submission


def train(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Main training routine.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for dataloaders.
        learning_rate (float): Peak learning rate for OneCycleLR.
        debug (bool): If True, runs on a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
    """
    # 1. Setup
    # Update Config based on arguments to ensure consistency across modules
    Config.EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = learning_rate
    Config.DEBUG = debug

    set_seed(Config.SEED)
    device = get_device()

    print(f"Starting experiment: {Config.EXPERIMENT_ID}")
    print(f"Device: {device}")
    print(f"Debug Mode: {debug}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Model Initialization
    # Determine input dimension from the dataset
    input_dim = train_loader.dataset.X.shape[-1]
    print(f"Input Dimension: {input_dim}")

    model = VentilatorModel(input_dim=input_dim).to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_mae = float("inf")

    for epoch in range(epochs):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss, val_mae = validate(model, val_loader, device)

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train MAE: {train_mae} | "
            f"Val Loss: {val_loss} | Val MAE: {val_mae}"
        )

        # Checkpoint
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            print(f"  >>> New Best MAE: {best_mae} saved.")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_mae,
            },
            is_best=is_best,
            filename="model.pth",
        )

    print(f"Training complete. Best Val MAE: {best_mae}")

    # 6. Submission
    generate_submission(model, test_loader, device)
