import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, get_device, calculate_rmse, print_metric
from library.dataset import DenoisingDataset
from library.model import CSKResUNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for i, (noisy_imgs, clean_imgs, _) in enumerate(dataloader):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # The model predicts the noise residual: Noise = Noisy - Clean
        noise_target = noisy_imgs - clean_imgs

        optimizer.zero_grad()

        # Forward pass
        noise_pred = model(noisy_imgs)

        # Calculate loss on the residual
        loss = criterion(noise_pred, noise_target)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Runs validation on the dataset and calculates RMSE.
    """
    model.eval()
    total_rmse = 0.0
    num_samples = 0

    with torch.no_grad():
        for noisy_imgs, clean_imgs, _ in dataloader:
            noisy_imgs = noisy_imgs.to(device)
            # clean_imgs are kept on CPU/Device as needed for metric calculation

            # Predict noise
            noise_pred = model(noisy_imgs)

            # Reconstruct clean image: Clean = Noisy - Predicted_Noise
            clean_pred = noisy_imgs - noise_pred

            # Clamp values to valid pixel range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Calculate RMSE
            # We calculate batch-wise RMSE and aggregate
            # Note: clean_imgs is a tensor from the dataloader
            batch_rmse = calculate_rmse(clean_imgs, clean_pred)

            batch_size = noisy_imgs.size(0)
            total_rmse += batch_rmse * batch_size
            num_samples += batch_size

    avg_rmse = total_rmse / num_samples if num_samples > 0 else 0.0
    return avg_rmse


def run_training():
    """
    Main function to set up and run the training process.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing datasets...")
    limit_size = Config.DEBUG_SUBSET_SIZE if Config.DEBUG else None

    train_dataset = DenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        root_dir=Config.INPUT_DIR,
        mode="train",
        load_cached_data=True,
        limit_size=limit_size,
    )

    val_dataset = DenoisingDataset(
        metadata_path=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        mode="val",
        load_cached_data=True,
        limit_size=limit_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Validation uses batch_size=1 to handle full resolution images
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"Training samples (patches): {len(train_dataset)}")
    print(f"Validation samples (images): {len(val_dataset)}")

    # 3. Model Initialization
    print("Initializing model...")
    model = CSKResUNet().to(device)

    # 4. Optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_rmse = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    start_time = time.time()

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_rmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Logging
        # Printing full precision for metrics
        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val RMSE: {val_rmse}"
        )

        # Checkpointing & Early Stopping
        if val_rmse < best_rmse:
            print(
                f"Validation RMSE improved from {best_rmse} to {val_rmse}. Saving model..."
            )
            best_rmse = val_rmse
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
    print_metric("Best Validation RMSE", best_rmse)
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
