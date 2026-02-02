import torch
import numpy as np
import os
from library.config import Config
from library.utils import mixup_data, mixup_criterion, get_score


def train_one_epoch(
    model, loader, optimizer, criterion, device, mixup_alpha=Config.MIXUP_ALPHA
):
    """
    Performs one epoch of training with Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, mixup_alpha, device
        )

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)

        # Compute Mixup loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Performs validation on the given loader and calculates AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    valid_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())
            valid_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate predictions and targets
    preds = np.concatenate(preds)
    valid_targets = np.concatenate(valid_targets)

    # Calculate AUC
    # Handle edge case where only one class is present in validation batch if necessary,
    # though get_score usually handles this or throws error which we let bubble up or handle in utils.
    auc = get_score(valid_targets, preds)

    return epoch_loss, auc


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
    save_path,
    patience=5,
):
    """
    Executes the full training loop with Early Stopping and Scheduler stepping.
    """
    best_val_auc = 0.0
    patience_counter = 0

    # Ensure directory exists for saving the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Log metrics with full precision
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"Model Saved! New Best AUC: {best_val_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_val_auc
