import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.utils import compute_mcc


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt is the probability of being classified correctly

        # Calculate alpha_t
        # If target=1, alpha_t = alpha. If target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Unpack batch
        # Dataset returns: X_kin, X_vis, X_cat, y
        x_kin, x_vis, x_cat, y = batch

        x_kin = x_kin.to(device)
        x_vis = x_vis.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device).view(
            -1, 1
        )  # Ensure target shape matches logit shape [Batch, 1]

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_kin, x_vis, x_cat)

        # Compute loss
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_kin.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, true labels, and predicted probabilities.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            x_kin, x_vis, x_cat, y = batch

            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device).view(-1, 1)

            logits = model(x_kin, x_vis, x_cat)
            loss = criterion(logits, y)

            running_loss += loss.item() * x_kin.size(0)

            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets).flatten()
    all_probs = np.vstack(all_probs).flatten()

    return total_loss, all_targets, all_probs


def optimize_threshold(y_true, y_probs):
    """
    Finds the best threshold that maximizes MCC on the validation set.
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Grid search from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, 0.01)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    device,
    epochs=Config.EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Main training loop with early stopping and threshold optimization.
    """
    criterion = FocalLoss(alpha=Config.FOCAL_LOSS_ALPHA, gamma=Config.FOCAL_LOSS_GAMMA)

    best_val_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_targets, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        # Optimize Threshold
        current_threshold, current_mcc = optimize_threshold(val_targets, val_probs)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.10f}")
        print(f"  Val Loss:   {val_loss:.10f}")
        print(
            f"  Val MCC:    {current_mcc:.10f} (at threshold {current_threshold:.4f})"
        )

        # Early Stopping Check
        if current_mcc > best_val_mcc:
            best_val_mcc = current_mcc
            best_threshold = current_threshold
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved!")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Val MCC: {best_val_mcc:.10f} at Threshold: {best_threshold:.4f}"
    )

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_threshold
