import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import AverageMeter, loss_fn, laplace_log_likelihood, seed_everything
from library.data import get_dataloaders
from library.model import PGARNet


def train_one_epoch(train_loader, model, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        # Move inputs to device
        axial = inputs["axial"].to(device)
        coronal = inputs["coronal"].to(device)
        tabular = inputs["tabular"].to(device)
        dt = inputs["dt"].to(device)
        base_fvc = inputs["base_fvc"].to(device)
        target = target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # The model returns fvc_pred, sigma_pred when dt/base_fvc are provided
        fvc_pred, sigma_pred = model(axial, coronal, tabular, dt, base_fvc)

        # Compute loss
        loss = loss_fn(target, fvc_pred, sigma_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), axial.size(0))

    return losses.avg


def validate(val_loader, model, device):
    """
    Evaluates the model on the validation set.
    Returns the average modified Laplace Log Likelihood metric.
    """
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            # Move inputs to device
            axial = inputs["axial"].to(device)
            coronal = inputs["coronal"].to(device)
            tabular = inputs["tabular"].to(device)
            dt = inputs["dt"].to(device)
            base_fvc = inputs["base_fvc"].to(device)
            target = target.to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(axial, coronal, tabular, dt, base_fvc)

            # Compute metric
            metric = laplace_log_likelihood(target, fvc_pred, sigma_pred)

            # Update metrics
            scores.update(metric.item(), axial.size(0))

    return scores.avg


def run_training(
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    load_cached_data=True,
):
    """
    Orchestrates the training process.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training with device: {device}")
    print(
        f"Hyperparameters: Epochs={epochs}, LR={lr}, WD={weight_decay}, Patience={patience}"
    )

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing PGAR-Net model...")
    model = PGARNet()
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training Loop
    best_score = -float("inf")
    early_stop_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training loop...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, device)

        # Validate
        val_score = validate(val_loader, model, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        # Note: Metric is negative, higher is better (closer to 0)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Metric: {val_score}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_score:
            print(f"New best score! ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            print("Early stopping triggered. Training finished.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_score
