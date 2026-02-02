import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import compute_robust_roc_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates loss for mixed inputs"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if Config.USE_MIXUP:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and robust ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    epoch_auc = compute_robust_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def fit_model(model, train_loader, val_loader, fold_idx, model_name):
    """
    Fits a specific model for a specific fold with early stopping.

    Args:
        model: The PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        fold_idx: Integer index of the current fold.
        model_name: String name of the model architecture.

    Returns:
        float: The best validation ROC AUC score achieved.
    """
    device = Config.DEVICE
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Monotonic Cosine Annealing (no restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    best_score = -float("inf")
    best_model_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_idx}_best.pth"
    )

    patience_counter = 0

    print(f"Starting training for {model_name} - Fold {fold_idx}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc} | "  # Full precision printing
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping Logic
        if val_auc > best_score:
            best_score = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Score! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best weights
    print(f"Loading best weights from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return best_score
