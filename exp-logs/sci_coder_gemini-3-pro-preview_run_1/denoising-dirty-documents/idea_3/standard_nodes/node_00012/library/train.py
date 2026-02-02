import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.dataset import DenoisingDataset
from library.model import AttentionUNet
from library.utils import calculate_rmse, print_metric, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(noisy)
        preds = torch.sigmoid(logits)

        # Calculate loss
        loss = criterion(preds, clean)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average Loss and average RMSE.
    """
    model.eval()
    running_loss = 0.0
    running_rmse = 0.0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            logits = model(noisy)
            preds = torch.sigmoid(logits)

            loss = criterion(preds, clean)
            running_loss += loss.item() * noisy.size(0)

            # Calculate RMSE for this image
            # Note: Batch size is 1 for validation
            batch_rmse = calculate_rmse(clean, preds)
            running_rmse += batch_rmse * noisy.size(0)

    total_samples = len(loader.dataset)
    avg_loss = running_loss / total_samples
    avg_rmse = running_rmse / total_samples

    return avg_loss, avg_rmse


def train_single_model(model_index, debug=False):
    """
    Trains a single instance of the Attention U-Net.
    """
    # 1. Set unique seed for this model in the ensemble
    current_seed = Config.SEED + model_index
    seed_everything(current_seed)

    print(f"\n{'='*40}")
    print(f"Training Model {model_index+1}/{Config.NUM_MODELS} (Seed: {current_seed})")
    print(f"{'='*40}")

    # 2. Prepare Data
    # Train loader: shuffled, batched, cropped patches
    train_dataset = DenoisingDataset(split="train", debug=debug)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val loader: sequential, batch_size=1 (variable image sizes), full images
    val_dataset = DenoisingDataset(split="val", debug=debug)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model, Loss, Optimizer, Scheduler
    device = torch.device(Config.DEVICE)
    model = AttentionUNet(n_channels=1, n_classes=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Cosine Annealing with decoupled horizon
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.COSINE_T_MAX, eta_min=1e-6
    )

    # 4. Training Loop with Early Stopping
    best_rmse = float("inf")
    patience_counter = 0

    # Adjust epochs if debugging
    num_epochs = 2 if debug else Config.NUM_EPOCHS

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging
        # Only print every few epochs or if best to reduce clutter,
        # but prompt asks to print metrics. We'll print every epoch.
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
        print_metric(f"Val RMSE (Model {model_index})", val_rmse)

        # Checkpoint & Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            filename = f"model_{model_index}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, filename)
            print(f"New best model saved with RMSE: {best_rmse:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Finished training Model {model_index}. Best RMSE: {best_rmse:.6f}")


def run_training(debug=False):
    """
    Orchestrates the training of the ensemble.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for i in range(Config.NUM_MODELS):
        train_single_model(model_index=i, debug=debug)
