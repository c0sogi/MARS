import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import sys

from library.config import Config
from library.utils import get_logger, save_model

# Initialize logger
logger = get_logger(name="engine")


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        criterion (loss_fn): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    avg_loss = running_loss / dataset_size
    logger.info(f"Training Loss: {avg_loss}")

    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        criterion (loss_fn): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply softmax to get probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate Macro ROC AUC
    # targets are likely one-hot or soft labels. roc_auc_score handles this with multi_class='ovr'
    try:
        auc = roc_auc_score(all_targets, all_preds, average="macro", multi_class="ovr")
    except ValueError as e:
        logger.warning(
            f"AUC calculation failed (likely due to single class in batch): {e}"
        )
        auc = 0.0

    # Print full precision as requested
    logger.info(f"Validation Loss: {avg_loss}")
    logger.info(f"Validation ROC AUC: {auc}")

    return avg_loss, auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (torch.nn.Module): The model to use for inference.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (image_ids, probabilities)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(image_ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_ids, all_preds


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=5,
    save_path=None,
):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (can be None for production mode).
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Scheduler.
        device: Device.
        num_epochs: Max epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    best_auc = -1.0
    epochs_no_improve = 0

    # If no validation set (Production mode), we just train for num_epochs
    use_validation = val_loader is not None

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if use_validation:
            val_loss, val_auc = evaluate(model, val_loader, criterion, device)

            # Scheduler step (usually based on epoch)
            if scheduler:
                # CosineAnnealingWarmRestarts expects step(epoch + batch_idx / len(loader))
                # or step() at end of epoch. Config says T_0 = MAX_EPOCHS, so step() per epoch is fine.
                scheduler.step()

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                if save_path:
                    logger.info(
                        f"Validation AUC improved to {best_auc}. Saving model..."
                    )
                    save_model(model, save_path)
            else:
                epochs_no_improve += 1
                logger.info(
                    f"No improvement in AUC. Patience: {epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break
        else:
            # Production mode: No validation, just save at the end or every epoch
            # Scheduler step
            if scheduler:
                scheduler.step()

            # In production, we assume the epoch count is already optimized (E_opt)
            # So we save the model at the end of the run.
            pass

    if not use_validation and save_path:
        logger.info("Training finished (Production mode). Saving final model...")
        save_model(model, save_path)
    elif use_validation:
        logger.info(f"Training finished. Best Validation AUC: {best_auc}")


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions and saves them to a CSV file in the submission format.

    Args:
        model: Trained model.
        dataloader: Test DataLoader.
        device: Device.
        output_path: Path to save the CSV.
    """
    logger.info("Generating submission...")
    ids, preds = predict(model, dataloader, device)

    # Create DataFrame
    # Columns: image_id, healthy, multiple_diseases, rust, scab
    # We assume Config.TARGET_COLS order matches the model output order
    # (which is determined by sorting or metadata order in data.py)

    df_sub = pd.DataFrame({"image_id": ids})

    for i, col in enumerate(Config.TARGET_COLS):
        df_sub[col] = preds[:, i]

    # Save
    df_sub.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
