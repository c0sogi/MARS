import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, MetricMonitor
from library.data import get_dataloaders
from library.model import GCRNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    monitor = MetricMonitor()

    for batch in loader:
        # Move inputs to device
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(images, tabular)

        # Calculate loss
        loss = criterion(preds, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        monitor.update(loss.item(), preds, target)

    return monitor.get_avg_loss(), monitor.get_avg_score()


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    monitor = MetricMonitor()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            preds = model(images, tabular)

            loss = criterion(preds, target)

            monitor.update(loss.item(), preds, target)

    return monitor.get_avg_loss(), monitor.get_avg_score()


def run_training():
    """
    Orchestrates the entire training process.
    """
    seed_everything(Config.SEED)
    device = Config.get_device()

    print(f"Using device: {device}")

    # Initialize Model
    model = GCRNet()
    model = model.to(device)

    # Differential Learning Rates
    # Filter parameters based on requires_grad (handled in model __init__)
    # Group parameters: Backbone vs Head/MLP
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ]
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # Loss Function
    criterion = LaplaceLogLikelihoodLoss()

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE)

    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_score = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_score = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full precision)
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss} | Train Score: {train_score}")
        print(f"Val Loss: {val_loss} | Val Score: {val_score}")

        # Save Best Model
        # Metric is negative Laplace Log Likelihood, higher is better (closer to 0)
        if val_score > best_score:
            print(f"New best score! Improving from {best_score} to {val_score}")
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved to {best_model_path}")

        print("-" * 30)

    print(f"Training complete. Best Validation Score: {best_score}")
