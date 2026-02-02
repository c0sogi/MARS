import os
import numpy as np
import torch
import torch.nn as nn
from library.utils import calculate_metrics


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss as a linear combination of losses for the two targets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, alpha=0.4):
    """
    Trains the model for one epoch using Mild Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        x = batch["image"].to(device, dtype=torch.float32)
        y = batch["targets"].to(device, dtype=torch.float32)
        batch_size = x.size(0)

        # Apply Mixup
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha, device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_x)

        # Compute loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set without augmentations.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, dtype=torch.float32)
            y = batch["targets"].to(device, dtype=torch.float32)
            batch_size = x.size(0)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate AUC
    auc_score = calculate_metrics(all_targets, all_preds)

    return epoch_loss, auc_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=50,
    patience=10,
    save_path=None,
):
    """
    Full training loop with Early Stopping and Best Model Saving.
    Calculates pos_weight dynamically based on training data balance.
    """
    # 1. Calculate pos_weight for BCEWithLogitsLoss
    # Access the underlying DataFrame from the dataset to compute class imbalance
    if hasattr(train_loader.dataset, "df") and hasattr(
        train_loader.dataset, "label_cols"
    ):
        df = train_loader.dataset.df
        label_cols = train_loader.dataset.label_cols
        targets = df[label_cols].values

        # Count positives and negatives per class
        pos_counts = np.sum(targets, axis=0)
        total_samples = len(targets)
        neg_counts = total_samples - pos_counts

        # Avoid division by zero
        pos_counts = np.maximum(pos_counts, 1)

        # Calculate weights: number_neg / number_pos
        pos_weight_val = neg_counts / pos_counts
        pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
    else:
        # Fallback if dataset structure is unexpected
        print(
            "Warning: Could not calculate pos_weight dynamically. Using default weights."
        )
        pos_weight = None

    # 2. Initialize Loss Function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, alpha=0.4
        )

        # Validation Step
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0

            # Save best model to disk immediately
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_auc
