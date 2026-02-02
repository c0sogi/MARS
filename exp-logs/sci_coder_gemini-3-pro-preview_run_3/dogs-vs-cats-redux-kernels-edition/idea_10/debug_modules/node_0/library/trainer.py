import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np

from library.config import Config
from library.utils import get_logger, calculate_log_loss, seed_everything

# Initialize Logger
logger = get_logger("trainer")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: [batch, 1]

        optimizer.zero_grad()

        # Mixed precision could be added here, but sticking to float32 for stability/simplicity
        # given the constraints and hardware.
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples
    duration = time.time() - start_time

    logger.info(
        f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {epoch_loss:.6f} - Time: {duration:.2f}s"
    )
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    avg_loss = running_loss / num_samples

    # Calculate metric using library utility
    metric_score = calculate_log_loss(targets, preds)

    return avg_loss, metric_score, np.array(preds)


def run_swa_training(model, train_loader, val_loader, model_name):
    """
    Runs the full training pipeline with SWA.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        model_name (str): Name of the model (for saving files).
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Standard Scheduler (Cosine Annealing) for the initial phase
    # We schedule it to run for the full epochs, but we might override it in SWA phase
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    best_metric = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

    logger.info(f"Starting training for {model_name} on {device}")
    logger.info(f"SWA Enabled: {Config.USE_SWA}, Start Epoch: {Config.SWA_START_EPOCH}")

    for epoch in range(Config.NUM_EPOCHS):
        # Check if we are in SWA phase
        in_swa_phase = Config.USE_SWA and epoch >= Config.SWA_START_EPOCH

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        if in_swa_phase:
            # SWA Phase: Update averaged model and use SWA scheduler
            logger.info(f"Epoch {epoch+1}: Updating SWA model parameters.")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # Standard Phase: Use standard scheduler
            scheduler.step()

        # Validate
        # Note: During SWA phase, we still validate the base model to monitor progress.
        # Validating swa_model requires update_bn which is expensive to do every epoch.
        val_loss, val_metric, _ = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch+1} - Val Loss: {val_loss:.10f} - Val LogLoss: {val_metric:.10f}"
        )

        # Save best model (Standard Phase logic)
        # Even in SWA phase, if the base model happens to be excellent, we track it.
        if val_metric < best_metric:
            best_metric = val_metric
            logger.info(f"New Best LogLoss: {best_metric:.10f}. Saving model...")
            torch.save(model.state_dict(), best_model_path)

    # --- End of Training Loop ---

    if Config.USE_SWA:
        logger.info("Training loop finished. Finalizing SWA model...")

        # Update Batch Normalization statistics for the averaged model
        # This is crucial because AveragedModel weights are averages, but BN stats need to be re-computed
        logger.info("Updating SWA Batch Norm statistics (this may take a while)...")
        update_bn(train_loader, swa_model, device=device)

        # Validate SWA Model
        swa_val_loss, swa_val_metric, _ = validate(
            swa_model, val_loader, criterion, device
        )
        logger.info(
            f"SWA Model - Val Loss: {swa_val_loss:.10f} - Val LogLoss: {swa_val_metric:.10f}"
        )

        # Save SWA Model
        swa_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_swa.pth")
        torch.save(swa_model.state_dict(), swa_model_path)
        logger.info(f"Saved SWA model to {swa_model_path}")

        # Return the SWA model as the final result
        return swa_model

    else:
        logger.info("Training finished (No SWA). Loading best model.")
        # Load best weights
        model.load_state_dict(torch.load(best_model_path))
        return model
