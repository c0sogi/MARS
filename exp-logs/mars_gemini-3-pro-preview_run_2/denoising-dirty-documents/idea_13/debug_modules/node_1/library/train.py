import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_rmse, save_checkpoint
from library.model import CoRes2NetUNet
from library.dataset import DenoisingDataset


def train_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Target is the noise residual
        target_noise = noisy - clean

        optimizer.zero_grad()

        # Model predicts noise
        predicted_noise = model(noisy)

        loss = criterion(predicted_noise, target_noise)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Validates the model on the validation set using RMSE.
    Performs full-image inference.
    """
    model.eval()
    rmse_scores = []

    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Model predicts noise
            predicted_noise = model(noisy)

            # Reconstruct clean image: Clean = Noisy - Noise
            predicted_clean = noisy - predicted_noise

            # Clamp to valid range [0, 1]
            predicted_clean = torch.clamp(predicted_clean, 0.0, 1.0)

            # Calculate RMSE for this image
            # calculate_rmse handles cpu/numpy conversion
            score = calculate_rmse(predicted_clean, clean)
            rmse_scores.append(score)

    avg_rmse = np.mean(rmse_scores)
    return avg_rmse


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=15,
):
    """
    Main function to run the training pipeline.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting training on device: {device}")

    # --- Data Loading ---
    # Training dataset (Patches)
    train_dataset = DenoisingDataset(mode="train", load_cached_data=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation dataset (Full Images)
    val_dataset = DenoisingDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one full image at a time
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples (patches): {len(train_dataset)}")
    print(f"Validation samples (images): {len(val_dataset)}")

    # --- Model Setup ---
    model = CoRes2NetUNet().to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # --- Training Loop ---
    best_score = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val RMSE: {val_score} | "  # Full precision printing
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpoint & Early Stopping
        if val_score < best_score:
            print(f"New best score! {best_score} -> {val_score}")
            best_score = val_score
            epochs_no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_score,
                Config.MODEL_CHECKPOINT_PATH,
            )
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time / 60:.2f} minutes.")
    print(f"Best Validation RMSE: {best_score}")

    return best_score
