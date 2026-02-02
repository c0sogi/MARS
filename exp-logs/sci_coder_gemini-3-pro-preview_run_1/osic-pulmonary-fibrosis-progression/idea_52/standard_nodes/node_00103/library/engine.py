import torch
import numpy as np
import sys
from library.config import Config
from library.utils import AverageMeter, compute_metric, get_logger, save_checkpoint
from library.loss import LaplaceLogLikelihoodLoss

# Initialize logger
logger = get_logger()


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Iterate over the dataloader
    # Note: No progress bar as per requirements
    for i, batch in enumerate(train_loader):
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        delta_week = batch["delta_week"].to(device)
        fvc_target = batch["fvc_target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred_fvc, pred_sigma = model(axial, coronal, tabular, base_fvc, delta_week)

        # Calculate loss
        loss = criterion(pred_fvc, pred_sigma, fvc_target)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update loss meter
        loss_meter.update(loss.item(), axial.size(0))

    logger.info(f"Epoch [{epoch}] Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes the Laplace Log Likelihood metric.
    """
    model.eval()
    loss_meter = AverageMeter()

    # Lists to store predictions and targets for global metric calculation
    all_true_fvc = []
    all_pred_fvc = []
    all_pred_sigma = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            fvc_target = batch["fvc_target"].to(device)

            # Forward pass
            pred_fvc, pred_sigma = model(axial, coronal, tabular, base_fvc, delta_week)

            # Calculate loss for tracking
            loss = criterion(pred_fvc, pred_sigma, fvc_target)
            loss_meter.update(loss.item(), axial.size(0))

            # Store predictions and targets (move to CPU numpy)
            all_true_fvc.append(fvc_target.cpu().numpy())
            all_pred_fvc.append(pred_fvc.cpu().numpy())
            all_pred_sigma.append(pred_sigma.cpu().numpy())

    # Concatenate all batches
    all_true_fvc = np.concatenate(all_true_fvc)
    all_pred_fvc = np.concatenate(all_pred_fvc)
    all_pred_sigma = np.concatenate(all_pred_sigma)

    # Compute metric
    # Metric values are negative and higher is better
    score = compute_metric(all_true_fvc, all_pred_fvc, all_pred_sigma)

    logger.info(f"Validation Loss: {loss_meter.avg}")
    # Print full precision as requested
    logger.info(f"Validation Metric Score: {score}")

    return loss_meter.avg, score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    criterion = LaplaceLogLikelihoodLoss()

    best_score = -float("inf")
    patience_counter = 0
    best_epoch = 0

    logger.info(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # 1. Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device
        )

        # 2. Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # 3. Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # 4. Save Checkpoint (Always save latest)
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_score": best_score,
            },
            is_best=False,
            filename="checkpoint.pth",
        )

        # 5. Early Stopping Logic
        # Metric is negative, higher is better (e.g., -6.5 is better than -7.0)
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0

            logger.info(
                f"New best score: {best_score} found at epoch {epoch}. Saving model..."
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "best_score": best_score,
                },
                is_best=True,
                filename=f"epoch_{epoch}.pth",
            )
        else:
            patience_counter += 1
            logger.info(
                f"Score did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            logger.info(
                f"Early stopping triggered. Best score was {best_score} at epoch {best_epoch}."
            )
            break

    logger.info("Training complete.")
    return best_score
