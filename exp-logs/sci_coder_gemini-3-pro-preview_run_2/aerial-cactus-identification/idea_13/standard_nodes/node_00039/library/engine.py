import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): The optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate ROC AUC
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        # Handle case where only one class is present in batch (unlikely for full val set)
        auc_score = 0.5

    return total_loss, auc_score


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, seed):
    """
    Main training loop with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device.
        seed (int): Current seed (for saving filenames).

    Returns:
        float: Best validation AUC score achieved.
    """
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for Seed {seed}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc = evaluate(model, val_loader, device)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save best model
            checkpoint_path = f"{Config.WORKING_DIR}/model_seed_{seed}.pth"
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training finished for Seed {seed}. Best Val AUC: {best_auc}")
    return best_auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set, optionally using TTA.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test dataloader.
        device (str): Device.

    Returns:
        tuple: (ids, probabilities)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Check if dataloader returns (image, label) or just image
            # Based on dataset.py, test_loader returns just image if labels are None,
            # but test_metadata usually has placeholder labels.
            # dataset.py __getitem__ returns (image, label) if labels is not None.
            # _load_cached_data loads labels for test if they exist in metadata.
            # test_metadata.csv has 'has_cactus' column (placeholder).
            # So dataset returns (image, label). We ignore label.

            if isinstance(batch, (list, tuple)):
                images = batch[0]
                # batch[1] are labels/placeholders, ignore
            else:
                images = batch

            # Collect IDs if possible.
            # The dataloader doesn't yield IDs directly in the batch loop unless modified.
            # However, the dataset object has .ids attribute.
            # We will collect predictions in order and map to dataset.ids later.

            images = images.to(device)

            # 1. Original Prediction
            logits = model(images)
            probs = torch.sigmoid(logits)

            if Config.USE_TTA:
                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])  # Width is dim 3 (B, C, H, W)
                logits_h = model(images_h)
                probs_h = torch.sigmoid(logits_h)

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])  # Height is dim 2
                logits_v = model(images_v)
                probs_v = torch.sigmoid(logits_v)

                # Average probabilities
                probs = (probs + probs_h + probs_v) / 3.0

            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Retrieve IDs from dataset
    all_ids = dataloader.dataset.ids

    return all_ids, all_preds
