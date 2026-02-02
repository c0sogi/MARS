import torch
import numpy as np
from library.config import DEVICE
from library.utils import calculate_roc_auc, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        # BCEWithLogitsLoss expects labels to have shape (B, 1) matching output
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect for AUC calculation
        probs = torch.sigmoid(outputs)
        all_targets.extend(labels.detach().cpu().numpy())
        all_preds.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))

    return epoch_loss, epoch_auc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))

    return val_loss, val_auc


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model: The PyTorch model.
        dataloader: The test DataLoader.
        device: The device to run on.

    Returns:
        tuple: (ids_array, predictions_array)
    """
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, _, ids in dataloader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            img_hflip = torch.flip(images, [3])
            out_hflip = model(img_hflip)
            prob_hflip = torch.sigmoid(out_hflip)

            # 3. Vertical Flip (dim 2 is height)
            img_vflip = torch.flip(images, [2])
            out_vflip = model(img_vflip)
            prob_vflip = torch.sigmoid(out_vflip)

            # Average probabilities
            avg_prob = (prob_orig + prob_hflip + prob_vflip) / 3.0

            all_preds.extend(avg_prob.cpu().numpy().flatten())
            all_ids.extend(ids)

    return np.array(all_ids), np.array(all_preds)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    patience,
    save_path,
):
    """
    Main training loop with early stopping and checkpointing.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model checkpoint.

    Returns:
        float: The best validation AUC score achieved.
    """
    best_auc = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

        # Update scheduler (CosineAnnealingLR expects step per epoch)
        if scheduler is not None:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch}: Train Loss: {train_loss} Train AUC: {train_auc} Val Loss: {val_loss} Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. Best AUC: {best_auc} at epoch {best_epoch}"
            )
            break

    return best_auc
