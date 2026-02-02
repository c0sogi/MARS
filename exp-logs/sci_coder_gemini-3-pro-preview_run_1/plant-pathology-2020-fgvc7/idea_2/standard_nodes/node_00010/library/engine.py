import os
import torch
import numpy as np
from library.config import Config
from library.utils import calculate_roc_auc


def get_weighted_criterion(df, device):
    """
    Constructs a Weighted CrossEntropyLoss criterion based on class frequencies in the dataframe.

    Args:
        df (pd.DataFrame): The training dataframe containing target labels.
        device (str): The device to load the weights onto.

    Returns:
        torch.nn.CrossEntropyLoss: The weighted loss function.
    """
    # Determine class counts
    # We use 'stratify_label' if available, otherwise reconstruct from one-hot
    if "stratify_label" in df.columns:
        counts = df["stratify_label"].value_counts()
    else:
        # Fallback: assume columns match CLASS_LABELS are present
        counts = df[Config.CLASS_LABELS].idxmax(axis=1).value_counts()

    # Ensure counts are ordered according to Config.CLASS_LABELS indices
    class_counts = []
    for label in Config.CLASS_LABELS:
        # Default to 0 if class missing, though unlikely with stratified split
        class_counts.append(counts.get(label, 0))

    class_counts = np.array(class_counts)
    total_samples = class_counts.sum()
    n_classes = len(class_counts)

    # Calculate weights: w_j = N / (n_classes * count_j)
    # Add epsilon to avoid division by zero
    weights = total_samples / (n_classes * (class_counts + 1e-6))

    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    return torch.nn.CrossEntropyLoss(weight=weights_tensor)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one epoch of training.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        tuple: (average_loss, average_auc)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * images.size(0)

        # Collect predictions for AUC calculation
        probs = torch.softmax(outputs, dim=1)
        all_targets.append(labels.cpu().numpy())
        all_preds.append(probs.detach().cpu().numpy())

    # Calculate epoch metrics
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (str): Device to run on.

    Returns:
        tuple: (average_loss, average_auc)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    # Calculate metrics
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    avg_loss = running_loss / len(dataloader.dataset)
    avg_auc = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, avg_auc


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    save_path,
    scheduler=None,
):
    """
    Orchestrates the training loop with Early Stopping and Model Checkpointing.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device string.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: The best validation AUC achieved.
    """
    best_val_auc = -1.0
    patience_counter = 0

    print(
        f"Starting training on {device} for {epochs} epochs with patience {patience}..."
    )

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print full precision metrics as required
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing and Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation AUC improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_val_auc
