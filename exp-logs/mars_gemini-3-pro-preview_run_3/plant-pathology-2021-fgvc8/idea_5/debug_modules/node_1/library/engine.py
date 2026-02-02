import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import get_score


def train_one_epoch(model, loader, optimizer, device, epoch, ema_model=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (str): The device to run on.
        epoch (int): Current epoch number.
        ema_model (ModelEmaV2, optional): The EMA model wrapper.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for _, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()

        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        device (str): The device to run on.

    Returns:
        tuple: (average_loss, f1_score)
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for _, (images, targets, _) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu())
            targets_list.append(targets.cpu())

    epoch_loss = running_loss / dataset_size

    preds_all = torch.cat(preds_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)

    # Calculate F1 Score
    score = get_score(targets_all, preds_all, threshold=Config.CONF_THRESHOLD)

    return epoch_loss, score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    ema_model=None,
    patience=5,
):
    """
    Executes the full training loop with Early Stopping and Checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): The learning rate scheduler.
        device (str): The device to run on.
        epochs (int): Total number of epochs.
        ema_model (ModelEmaV2, optional): The EMA model wrapper.
        patience (int): Early stopping patience.
    """
    best_score = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, ema_model
        )

        # Validate (Primary Model)
        val_loss, val_score = validate(model, val_loader, device)

        # Validate (EMA Model) if available
        val_loss_ema = None
        val_score_ema = None

        # Determine which score to use for Early Stopping / Checkpointing
        current_score = val_score

        if ema_model is not None:
            val_loss_ema, val_score_ema = validate(ema_model.module, val_loader, device)
            # If EMA is used, we typically prefer its performance for the final model
            current_score = val_score_ema

        # Print Metrics (Full Precision)
        log_msg = f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val F1 = {val_score}"
        if ema_model is not None:
            log_msg += f", EMA Val Loss = {val_loss_ema}, EMA Val F1 = {val_score_ema}"
        print(log_msg)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Checkpointing & Early Stopping
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0

            # Save Best Model
            # If EMA is enabled, save the EMA weights as the best model
            save_dict = (
                ema_model.module.state_dict()
                if ema_model is not None
                else model.state_dict()
            )
            torch.save(save_dict, Config.BEST_MODEL_PATH)
            print(f"New best model saved with F1 Score: {best_score}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print(f"Training complete. Best F1 Score: {best_score}")
