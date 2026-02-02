import os
import time
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from library.config import Config
from library.dataset import IceCubeDataset
from library.model import DFCGN
from library.loss import CosineSimilarityLoss, get_angular_error


def set_seeds(seed):
    """
    Sets the random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred = model(batch)

        # Compute loss
        # batch.y is (Batch, 2) -> [azimuth, zenith]
        loss = criterion(pred, batch.y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Scheduler step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and average Mean Angular Error (MAE).
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            pred = model(batch)

            # Compute loss
            loss = criterion(pred, batch.y)

            # Compute metric (MAE)
            mae = get_angular_error(pred, batch.y)

            total_loss += loss.item()
            total_mae += mae
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_mae


def train_model():
    """
    Main function to train the DF-CGN model.
    """
    # 1. Setup
    set_seeds(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Datasets...")
    # Note: Dataset handles caching internally
    train_dataset = IceCubeDataset(mode="train")
    val_dataset = IceCubeDataset(mode="val")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Use PyG DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DFCGN().to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires total steps
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = CosineSimilarityLoss()

    # 5. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MAE: {val_mae}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation Loss: {best_val_loss}")
