import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.data import get_dataloaders
from library.model import ZIResDnCNN


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model outputs the predicted clean image
        outputs = model(inputs)

        # Loss calculation (MSE betweeen Predicted Clean and Target Clean)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
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
    return epoch_loss


def train_model(
    num_epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = ZIResDnCNN(
        num_blocks=Config.NUM_BLOCKS,
        num_channels=Config.NUM_CHANNELS,
        kernel_size=Config.KERNEL_SIZE,
        padding=Config.PADDING,
        use_zero_gamma=Config.USE_ZERO_GAMMA,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=Config.MIN_LEARNING_RATE
    )

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # Early Stopping Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_loss = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate RMSE
        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)

        # Log Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss} | Train RMSE: {train_rmse} | "
            f"Val Loss: {val_loss} | Val RMSE: {val_rmse} | "
            f"LR: {current_lr}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, filename=Config.MODEL_SAVE_PATH
            )
            print(f"New best model saved at epoch {epoch} with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            print(
                f"EarlyStopping counter: {patience_counter} out of {Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time} seconds.")
    print(f"Best Val Loss: {best_val_loss} at Epoch {best_epoch}")
