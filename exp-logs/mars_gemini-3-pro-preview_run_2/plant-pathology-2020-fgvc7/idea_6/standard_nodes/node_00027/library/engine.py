import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.utils import print_metric


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    criterion,
    cfg,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Cite solution_lesson_node_00002: Apply Label Smoothing
        if cfg.label_smoothing > 0:
            targets = targets * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss_scaled = loss / cfg.gradient_accumulation_steps
        loss_scaled.backward()

        if (i + 1) % cfg.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation dataloader.
        device (torch.device): Device to run on.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: (Average Loss, Mean Column-wise ROC AUC)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities for binary classification
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Mean Column-wise ROC AUC
    try:
        # average='macro' computes AUC for each label and takes the unweighted mean
        auc_score = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError:
        # Fallback if a class is missing in the batch/set (rare with stratified split)
        auc_score = 0.5

    return epoch_loss, auc_score


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, criterion, cfg, fold
):
    """
    Runs the training loop for a specific fold with Early Stopping.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: LR Scheduler.
        device: Device.
        criterion: Loss function.
        cfg: Config object.
        fold: Fold index (for saving).

    Returns:
        tuple: (Best Model, Best AUC)
    """
    best_auc = -1.0
    patience_counter = 0
    best_model_path = cfg.best_model_path_format.format(fold)

    # Ensure directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            cfg,
        )
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        # Print metrics with full precision
        print_metric(f"Fold {fold} | Epoch {epoch+1} | Train Loss", train_loss)
        print_metric(f"Fold {fold} | Epoch {epoch+1} | Val Loss", val_loss)
        print_metric(f"Fold {fold} | Epoch {epoch+1} | Val AUC", val_auc)

        # Scheduler Step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"Early stopping triggered for fold {fold} at epoch {epoch+1}")
            break

    # Load best model weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return model, best_auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: Trained model.
        dataloader: Test DataLoader.
        device: Device.

    Returns:
        tuple: (List of image_ids, Numpy array of probabilities)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    return all_ids, np.concatenate(all_preds, axis=0)
