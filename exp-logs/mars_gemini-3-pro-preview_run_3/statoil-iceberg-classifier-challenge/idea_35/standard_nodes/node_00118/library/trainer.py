import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from library.utils import set_seed, EarlyStopping
from library.model import DPACNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (Batch, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Convert logits to probabilities for accuracy calculation
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate accuracy
    preds_binary = (all_preds > 0.5).astype(int)
    val_acc = accuracy_score(all_labels, preds_binary)

    return val_loss, val_acc


def run_fold(
    train_loader,
    val_loader,
    fold_idx=0,
    epochs=75,
    patience=12,
    learning_rate=1e-3,
    weight_decay=1e-2,
    device=None,
    checkpoint_dir="./checkpoints",
):
    """
    Runs the training and evaluation loop for a single fold.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        fold_idx: Index of the current fold (for saving checkpoints).
        epochs: Maximum number of epochs.
        patience: Patience for early stopping.
        learning_rate: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        device: Torch device (cuda/cpu).
        checkpoint_dir: Directory to save model checkpoints.

    Returns:
        model: The trained model with best weights loaded.
        best_loss: The best validation loss achieved.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting training for Fold {fold_idx} on device: {device}")

    # Initialize Model
    model = DPACNN()
    model.to(device)

    # Optimizer: AdamW with constant learning rate
    # Decoupled weight decay is handled correctly by AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Loss Function: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx}.pth")
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=checkpoint_path
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss: {val_loss:.10f} - "
            f"Val Acc: {val_acc:.10f}"
        )

        # Check early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model weights
    print(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

    return model, early_stopping.best_score
