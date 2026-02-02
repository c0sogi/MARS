import torch
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, save_checkpoint, log_metrics, seed_everything
from library.loss import LaplaceNLLLoss
from library.data import get_dataloaders
from library.model import PGBBNet


def train_epoch(loader, model, criterion, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter("Loss", ":.4f")

    for batch_idx, batch in enumerate(loader):
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)

        # Forward pass
        preds = model(axial, coronal, tabular, meta)

        # Compute loss
        loss = criterion(preds, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), axial.size(0))

    return losses.avg


def validate_epoch(loader, model, criterion, device):
    """
    Performs evaluation on the validation set.
    """
    model.eval()
    losses = AverageMeter("Loss", ":.4f")

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            preds = model(axial, coronal, tabular, meta)
            loss = criterion(preds, target)

            losses.update(loss.item(), axial.size(0))

    return losses.avg


def train_model(debug=Config.DEBUG):
    """
    Main training routine.
    Initializes model, data, optimizer, and runs the training loop with early stopping.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # Initialize Model
    model = PGBBNet().to(device)

    # Initialize Loss
    criterion = LaplaceNLLLoss().to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training Loop Variables
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Debug Mode: {debug}")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate_epoch(val_loader, model, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Log Metrics
        # Note: The metric is negative loss. We log raw loss values.
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metric": -val_loss,  # Competition metric is negative NLL
        }
        log_metrics(epoch, metrics)

        # Checkpointing and Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            },
            is_best,
            checkpoint_dir=Config.WORKING_DIR,
        )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Training complete. Best Validation Loss: {best_loss}")
    return best_loss
