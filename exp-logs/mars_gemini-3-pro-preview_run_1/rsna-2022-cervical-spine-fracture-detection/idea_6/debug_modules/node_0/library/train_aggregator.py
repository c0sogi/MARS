import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library import config, models, datasets, utils


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedMultiLabelLogLoss(nn.Module):
    """
    Weighted multi-label logarithmic loss.
    L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    """

    def __init__(self, device):
        super().__init__()
        # Weights: 1.0 for C1-C7, 7.0 for patient_overall
        # This balances the single overall label against the 7 specific labels
        weights = [1.0] * 7 + [7.0]
        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.epsilon = 1e-7

    def forward(self, logits, targets):
        """
        logits: (B, 8) - raw scores (before sigmoid)
        targets: (B, 8) - binary labels 0 or 1
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Clamp for numerical stability
        probs = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)

        # Calculate binary log loss terms
        # term1: y * log(p)
        term1 = targets * torch.log(probs)
        # term2: (1-y) * log(1-p)
        term2 = (1.0 - targets) * torch.log(1.0 - probs)

        # Weighted sum
        # loss per class j: -w_j * (term1 + term2)
        loss_per_class = -self.weights * (term1 + term2)

        # Average across batch, then sum/mean across classes
        # The prompt says "loss is averaged across all rows".
        # Usually this means average over batch samples.
        # For columns, we can take the mean or sum.
        # Given the weights are explicit, we likely want the mean of the weighted losses per sample.

        return torch.mean(loss_per_class)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for visual_feats, anat_ids, labels in loader:
        visual_feats = visual_feats.to(device)
        anat_ids = anat_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(visual_feats, anat_ids)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * visual_feats.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for visual_feats, anat_ids, labels in loader:
            visual_feats = visual_feats.to(device)
            anat_ids = anat_ids.to(device)
            labels = labels.to(device)

            logits = model(visual_feats, anat_ids)
            loss = criterion(logits, labels)

            running_loss += loss.item() * visual_feats.size(0)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate AUC for patient_overall (last column, index 7)
    # Handle case where only one class is present in batch
    try:
        overall_auc = roc_auc_score(all_targets[:, 7], all_probs[:, 7])
    except ValueError:
        overall_auc = 0.5

    return epoch_loss, overall_auc


def run_stage3_training(
    epochs=config.STAGE3_CONFIG["epochs"],
    batch_size=config.STAGE3_CONFIG["batch_size"],
    lr=config.STAGE3_CONFIG["lr"],
    patience=5,
):
    set_seed()

    print(f"Starting Stage 3 Training: Anatomically-Grouped Recurrent Aggregator")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    # 1. Datasets
    # Ensure features exist or handle gracefully (SequenceDataset handles dummies)
    train_ds, val_ds = datasets.get_datasets(stage="stage3")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model
    model = models.AnatomicalBiGRU()
    model = model.to(config.DEVICE)

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    criterion = WeightedMultiLabelLogLoss(config.DEVICE)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "fracture_aggregator.pth")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, config.DEVICE)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Overall AUC: {val_auc}")

        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            print(
                f"Validation Loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Stage 3 Training Completed. Best Val Loss: {best_val_loss}")
    print(f"Model saved to: {checkpoint_path}")
