import torch
import torch.nn as nn
import numpy as np
import copy
from library.config import Config
from library.utils import calculate_roc_auc


def get_pos_weight(dataset, device):
    """
    Calculates positive weights for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = number_of_negatives / number_of_positives
    """
    # Access the labels from the dataset (N, Num_Classes)
    if hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        # Fallback if dataset is a Subset or wrapped
        # This assumes the underlying dataset has labels
        labels = dataset.dataset.labels[dataset.indices]

    pos_counts = np.sum(labels, axis=0)
    total_counts = len(labels)
    neg_counts = total_counts - pos_counts

    # Add epsilon to prevent division by zero
    weights = neg_counts / (pos_counts + 1e-6)

    return torch.as_tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one epoch of training with Mild Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        # Mild Mixup
        alpha = Config.MIXUP_ALPHA
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        index = torch.randperm(batch_size).to(device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

        # Forward pass
        outputs = model(mixed_inputs)

        # Mixup Loss
        # We use the same criterion (BCEWithLogitsLoss) for both parts
        loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(
            outputs, targets[index]
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Runs validation on the provided loader.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        epoch_auc = calculate_roc_auc(all_targets, all_preds)
    else:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    patience,
    scheduler=None,
):
    """
    Orchestrates the training process with Early Stopping and Scheduler.
    """
    # Calculate class weights for the loss function
    pos_weight = get_pos_weight(train_loader.dataset, device)

    # Initialize Loss Function with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print(f"Starting training for {num_epochs} epochs with patience {patience}...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Logic (Maximize AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return history
