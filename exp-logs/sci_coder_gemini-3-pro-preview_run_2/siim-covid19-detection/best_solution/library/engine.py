import math
import sys
import time
import torch
from library.config import Config
from library.utils import AverageMeter, log_metrics, save_checkpoint


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trains the model for one epoch.
    """
    model.train()

    # Initialize meters for all potential losses
    meters = {
        "loss": AverageMeter("Total Loss"),
        "loss_classifier": AverageMeter("Cls Loss"),
        "loss_box_reg": AverageMeter("Box Loss"),
        "loss_objectness": AverageMeter("Obj Loss"),
        "loss_rpn_box_reg": AverageMeter("RPN Box Loss"),
        "loss_global_classifier": AverageMeter("Global Loss"),
    }

    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())

        # Check for infinity
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update meters
        meters["loss"].update(loss_value)
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(v.item())

        # Logging
        if i % print_freq == 0:
            # Construct metrics dictionary for logging
            log_dict = {
                "Epoch": f"[{epoch}] [{i}/{len(data_loader)}]",
                "Time": f"{time.time() - start_time:.2f}s",
                "Loss": meters["loss"].avg,
            }
            # Add individual losses
            for k, m in meters.items():
                if k != "loss":
                    log_dict[k] = m.avg

            log_metrics(log_dict)

    return meters["loss"].avg


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """
    Evaluates the model on the validation set and returns the average loss.
    Note: Sets model to train mode to retrieve loss dict, but with no_grad.
    """
    # We must switch to train mode to get loss dictionary from FasterRCNN
    model.train()

    loss_meter = AverageMeter("Val Loss")

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_meter.update(losses.item())

    return loss_meter.avg


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=3,
):
    """
    Orchestrates the training process with early stopping and checkpointing.
    """
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss = evaluate_loss(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Log Epoch Metrics
        print(f"\nEnd of Epoch {epoch}")
        log_metrics(
            {
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "LR": optimizer.param_groups[0]["lr"],
            }
        )

        # Checkpointing
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save Checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            is_best,
            checkpoint_dir=Config.WORKING_DIR,
        )

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"\nEarly stopping triggered after {epoch + 1} epochs. Best Val Loss: {best_loss}"
            )
            break

    print(f"Training completed. Best Validation Loss: {best_loss}")
