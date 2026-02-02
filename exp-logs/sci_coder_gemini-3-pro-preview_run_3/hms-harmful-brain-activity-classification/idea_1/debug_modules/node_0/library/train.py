import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, save_checkpoint, AverageMeter, print_metrics
from library.data import get_dataloaders
from library.model import BiGRUModel


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): The training data loader.
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(data)

        # KLDivLoss expects log-probabilities as input
        # Apply log_softmax to logits
        log_probs = F.log_softmax(logits, dim=1)

        # Compute loss
        loss = criterion(log_probs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), data.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader (DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)

            logits = model(data)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets)

            losses.update(loss.item(), data.size(0))

    return losses.avg


def train_model():
    """
    Main training routine. Initializes data, model, optimizer, and runs the training loop.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # 3. Model
    model = BiGRUModel()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation loss plateaus
    # Using a patience slightly lower than early stopping patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=False
    )

    # Criterion: KL Divergence
    # reduction='batchmean' ensures mathematical correctness for KL Div over batches
    criterion = nn.KLDivLoss(reduction="batchmean")

    # 5. Training Loop
    best_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train & Validate
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )
        val_loss = validate(val_loader, model, criterion, device)

        # Adjust Learning Rate
        scheduler.step(val_loss)

        # Logging
        duration = time.time() - start_time
        metrics = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Time": duration,
        }
        print_metrics(metrics)

        # Checkpointing & Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "val_loss": val_loss,
            },
            is_best=is_best,
            filename="checkpoint.pth",
        )

        # Early Stopping
        if epochs_no_improve >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training finished. Best Validation Loss: {best_loss}")
