import os
import time
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_device, Logger
from library.dataset import prepare_data
from library.model import WideStateNet
from library.loss import CompositeMaskedL1Loss


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Performs one training epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        x = batch["x"].to(device)
        u_out = batch["u_out"].to(device)
        y = batch["y"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (returns tuple: final_pred, aux_pred)
        preds = model(x, u_out)

        # Compute composite loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Strict threshold of 1.0 for Wide-State LSTM stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIPPING)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step (OneCycleLR steps every batch)
        scheduler.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate_epoch(model, loader, criterion, device):
    """
    Performs validation. Returns the MAE of the final head on the inspiratory phase.
    """
    model.eval()
    running_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            # Forward pass
            preds = model(x, u_out)
            final_pred, _ = preds

            # We calculate the metric specifically for the final head
            # The criterion's masked_mae method is suitable for the raw metric
            # provided we pass just the final prediction.
            loss = criterion.masked_mae(final_pred, y, u_out)

            running_mae += loss.item()
            num_batches += 1

    return running_mae / num_batches


def train_model():
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = Logger("training_log.txt")

    logger.log(f"Starting experiment: {Config.EXPERIMENT_NAME}")
    logger.log(f"Device: {device}")
    logger.log(f"Batch Size: {Config.BATCH_SIZE}")
    logger.log(f"Epochs: {Config.EPOCHS}")

    # 2. Data
    logger.log("Preparing data...")
    train_loader, val_loader, _, feature_names = prepare_data(
        load_cached_data=Config.USE_CACHE
    )

    input_dim = len(feature_names)
    logger.log(f"Input Features ({input_dim}): {feature_names}")

    # 3. Model
    logger.log(
        "Initializing Wide-State Weight-Normalized Physics-Injected Composite Network..."
    )
    model = WideStateNet(input_dim=input_dim, feature_names=feature_names)
    model = model.to(device)

    # 4. Optimization
    criterion = CompositeMaskedL1Loss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR calculation
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
    )

    # 5. Training Loop
    best_val_mae = float("inf")
    model_save_path = os.path.join(Config.WORKING_DIR, "model.pth")

    logger.log("Starting training loop...")
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_mae = validate_epoch(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Log metrics
        # Printing full precision as requested
        logger.log(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val MAE: {val_mae} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpoint
        if val_mae < best_val_mae:
            logger.log(
                f"Validation MAE improved from {best_val_mae} to {val_mae}. Saving model..."
            )
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path)

    total_time = time.time() - start_time
    logger.log(f"Training complete. Total time: {total_time:.2f}s")
    logger.log(f"Best Validation MAE: {best_val_mae}")

    logger.close()
