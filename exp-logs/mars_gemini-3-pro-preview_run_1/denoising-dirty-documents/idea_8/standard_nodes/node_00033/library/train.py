import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import DEVICE, EPOCHS, LEARNING_RATE, WORKING_DIR
from library.utils import seed_everything, rmse_score, save_checkpoint
from library.model import ASPPUNet
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        # Accumulate loss (MSE is mean per pixel, so we weight by batch size)
        running_loss += loss.item() * noisy.size(0)
        count += noisy.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set and calculates global RMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Validation loader yields: noisy, clean, id
            noisy, clean, _ = batch
            noisy = noisy.to(device)

            outputs = model(noisy)

            # Collect flattened arrays for global RMSE calculation
            # clean is usually on CPU from loader, ensure numpy
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(clean.numpy().flatten())

    if not all_preds:
        return 0.0

    # Concatenate all pixels to compute true global RMSE
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return rmse_score(y_true, y_pred)


def train_model(seed, load_cached_data=True, patience=100):
    """
    Trains a single model instance.

    Args:
        seed (int): Random seed for reproducibility.
        load_cached_data (bool): Whether to attempt loading data from cache.
        patience (int): Epochs to wait for improvement before early stopping.

    Returns:
        float: The best validation RMSE achieved.
    """
    # 1. Set Seed
    seed_everything(seed)

    # 2. Load Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Initialize Model
    model = ASPPUNet().to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_rmse = float("inf")
    epochs_no_improve = 0
    save_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")

    print(f"Starting training for Seed {seed} on {DEVICE}...")

    for epoch in range(EPOCHS):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validation Step
        val_rmse = validate(model, val_loader, DEVICE)

        # Scheduler Step
        scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss} | Val RMSE: {val_rmse}"
        )

        # Checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_rmse": best_rmse,
                },
                save_path,
            )
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training finished for Seed {seed}. Best Val RMSE: {best_rmse}")
    return best_rmse
