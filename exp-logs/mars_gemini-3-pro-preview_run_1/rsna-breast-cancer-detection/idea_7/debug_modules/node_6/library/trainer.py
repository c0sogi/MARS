import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SpatialSiameseEfficientNet


def probabilistic_f1(y_true, y_pred_probs, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true (np.array): Ground truth binary labels (0 or 1).
        y_pred_probs (np.array): Predicted probabilities [0, 1].
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The pF1 score.
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    # pTP = Sum(prob * label)
    p_tp = np.sum(y_pred_probs * y_true)

    # pFP = Sum(prob * (1 - label))
    p_fp = np.sum(y_pred_probs * (1 - y_true))

    # Total Positives (TP + FN) in ground truth
    total_pos = np.sum(y_true)

    # pPrecision = pTP / (pTP + pFP)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_pos + epsilon)

    # pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return p_f1


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        # Unpack batch
        images = batch["image"].to(device, non_blocking=True)
        contra_images = batch["contra_image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad()

        # Forward pass (Siamese)
        logits = model(images, contra_images)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimization (No Gradient Clipping per requirements)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and pF1 score.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            contra_images = batch["contra_image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).view(-1, 1)

            logits = model(images, contra_images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()

    pf1 = probabilistic_f1(all_labels, all_probs)

    return total_loss, pf1


def run_training(debug=False, load_cached=True):
    """
    Main training pipeline.

    Args:
        debug (bool): If True, runs on a subset of data for fewer epochs.
        load_cached (bool): Whether to load processed data from cache.
    """
    seed_everything(Config.SEED)

    print(f"Starting training on device: {Config.DEVICE}")
    print(f"Debug Mode: {debug}")

    # 1. Data Loaders
    train_loader, val_loader, _ = get_dataloaders(load_cached=load_cached, debug=debug)

    # 2. Model
    model = SpatialSiameseEfficientNet()
    model.to(Config.DEVICE)

    # 3. Loss Function
    # Weighted BCE to handle 1:47 imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    num_epochs = 2 if debug else Config.NUM_EPOCHS

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val pF1: {val_pf1}"
        )

        # Early Stopping & Model Saving
        # We save based on best pF1 score
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with pF1: {best_pf1}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation pF1: {best_pf1}")
