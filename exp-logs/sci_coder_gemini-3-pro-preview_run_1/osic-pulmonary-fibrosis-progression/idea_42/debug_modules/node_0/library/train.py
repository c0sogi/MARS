import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import DEVICE, EPOCHS, PATIENCE, LEARNING_RATE, WORKING_DIR, SEED
from library.utils import seed_everything, AverageMeter, laplace_log_likelihood_loss
from library.data import get_dataloaders
from library.model import CRHDAN


def train_one_epoch(loader, model, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch
        # Structure from OSICDataset:
        # (axial, coronal, tab_vec, weeks_diff, base_fvc, target_fvc)
        axial = batch[0].to(device)
        coronal = batch[1].to(device)
        tab_vec = batch[2].to(device)
        weeks_diff = batch[3].to(device)
        base_fvc = batch[4].to(device)
        target_fvc = batch[5].to(device)

        # Forward pass
        pred_fvc, pred_sigma = model(axial, coronal, tab_vec, weeks_diff, base_fvc)

        # Compute loss
        loss = laplace_log_likelihood_loss(target_fvc, pred_fvc, pred_sigma)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), axial.size(0))

    return losses.avg


def evaluate(loader, model, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (negative metric).
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch[0].to(device)
            coronal = batch[1].to(device)
            tab_vec = batch[2].to(device)
            weeks_diff = batch[3].to(device)
            base_fvc = batch[4].to(device)
            target_fvc = batch[5].to(device)

            # Forward pass
            pred_fvc, pred_sigma = model(axial, coronal, tab_vec, weeks_diff, base_fvc)

            # Compute loss
            loss = laplace_log_likelihood_loss(target_fvc, pred_fvc, pred_sigma)

            losses.update(loss.item(), axial.size(0))

    return losses.avg


def train_model():
    """
    Main training loop with Early Stopping.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # Initialize DataLoaders
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize Model
    model = CRHDAN()
    model = model.to(DEVICE)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Training State
    best_val_loss = float("inf")
    early_stopping_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training on device: {DEVICE}")
    print(f"Total Epochs: {EPOCHS}, Patience: {PATIENCE}")

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, DEVICE)

        # Validate
        val_loss = evaluate(val_loader, model, DEVICE)

        # Update Scheduler
        scheduler.step()

        # The metric in the competition is negative loss (Higher is better).
        # Our loss function returns positive values (Lower is better).
        # We print both for clarity.
        val_metric = -val_loss

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Metric: {val_metric}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            early_stopping_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stopping_counter}/{PATIENCE}"
            )

        if early_stopping_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
    return best_model_path
