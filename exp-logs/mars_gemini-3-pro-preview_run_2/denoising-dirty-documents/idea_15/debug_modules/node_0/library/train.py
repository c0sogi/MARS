import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.model import RepCResUNetSR
from library.data_loader import get_dataloaders


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=15, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=15,
):
    """
    Main training function for Rep-CResUNet-SR.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        patience (int): Patience for early stopping.

    Returns:
        float: Best validation RMSE achieved.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loaders
    loaders = get_dataloaders(batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 3. Model Initialization
    model = RepCResUNetSR().to(device)

    # 4. Optimizer and Scheduler
    # Using AdamW with aggressive weight decay as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 5. Loss Function
    # Minimizing MSE on the output (Clean Prediction) vs Target (Clean Ground Truth)
    # This is mathematically equivalent to minimizing MSE on the noise residual.
    criterion = nn.MSELoss()

    # 6. Early Stopping
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    best_val_rmse = float("inf")

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # Model returns predicted clean image: x - noise_pred
            outputs = model(noisy_imgs)

            # Compute loss
            loss = criterion(outputs, clean_imgs)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy_imgs.size(0)

        # Average training loss
        train_loss = train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_rmse_accum = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for noisy_imgs, clean_imgs, _ in val_loader:
                noisy_imgs = noisy_imgs.to(device)

                # Forward pass (Full image)
                outputs = model(noisy_imgs)

                # Calculate RMSE for this image
                # Move to CPU for numpy calculation in utils
                rmse = calculate_rmse(clean_imgs, outputs)

                val_rmse_accum += rmse
                num_val_samples += 1

        avg_val_rmse = val_rmse_accum / num_val_samples

        # --- Logging ---
        end_time = time.time()
        epoch_duration = end_time - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss (MSE): {train_loss:.10f} | "
            f"Val RMSE: {avg_val_rmse:.10f}"
        )

        # --- Scheduler Step ---
        scheduler.step()

        # --- Checkpointing & Early Stopping ---
        # We use the EarlyStopping class to handle saving the best model
        early_stopping(avg_val_rmse, model, Config.BEST_MODEL_PATH)

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation RMSE: {best_val_rmse:.10f}")
    return best_val_rmse
