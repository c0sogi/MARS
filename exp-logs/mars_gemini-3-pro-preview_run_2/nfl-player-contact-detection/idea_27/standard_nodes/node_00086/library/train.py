import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
from library.config import (
    WORKING_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    CAT_COLS,
    EARLY_STOPPING_PATIENCE,
    SEED,
)
from library.utils import seed_everything, get_device, compute_mcc
from library.dataset import get_dataloaders
from library.model import SEARVN


class FocalLoss(nn.Module):
    """
    Focal Loss for dense binary classification.
    FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
    Implemented using BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, 1] logits
        # targets: [N, 1] binary targets

        # Calculate standard BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # p = sigmoid(inputs)
        # if y=1, p_t = p; if y=0, p_t = 1-p
        # This is equivalent to exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        # if y=1, alpha_t = alpha; if y=0, alpha_t = 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_vocab_sizes():
    """
    Loads the fitted encoders to determine vocabulary sizes for embeddings.
    Must be called after DataProcessor has run.
    """
    path = os.path.join(WORKING_DIR, "encoders.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Encoders file not found at {path}. Ensure data processing is complete."
        )

    encoders = joblib.load(path)
    vocab_sizes = {}
    for col in CAT_COLS:
        # The embedding layer needs size = num_classes
        # DataProcessor handles unknown mapping, so classes_ covers the vocab
        vocab_sizes[col] = len(encoders[col].classes_)
    return vocab_sizes


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        x_kin = batch["X_kin"].to(device)
        x_vis = batch["X_vis"].to(device)
        x_cat = batch["X_cat"].to(device)
        y = batch["y"].to(device).unsqueeze(1)  # Ensure [Batch, 1] shape

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_kin, x_vis, x_cat)

        # Loss calculation
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    """
    Performs validation, calculating Loss and MCC.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_kin = batch["X_kin"].to(device)
            x_vis = batch["X_vis"].to(device)
            x_cat = batch["X_cat"].to(device)
            y = batch["y"].to(device).unsqueeze(1)

            logits = model(x_kin, x_vis, x_cat)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)

            # Predictions for MCC (using 0.5 threshold for monitoring)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute Metrics
    mcc = compute_mcc(all_targets, all_preds)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, mcc


def train_model(debug=False, load_cached_data=True):
    """
    Main function to orchestrate model training.
    """
    seed_everything(SEED)
    device = get_device()
    print(f"Device: {device}")

    # 1. Load Data
    # This triggers DataProcessor, which creates the cache and encoders.joblib
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        debug=debug, load_cached_data=load_cached_data
    )

    # 2. Get Vocab Sizes
    # Must happen after get_dataloaders
    vocab_sizes = get_vocab_sizes()

    # 3. Initialize Model
    print("Initializing SEARVN model...")
    model = SEARVN(vocab_sizes=vocab_sizes).to(device)

    # 4. Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    # 5. Training Loop
    best_mcc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mcc = validate_epoch(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MCC: {val_mcc}"
        )

        # Early Stopping Logic
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print("Saved new best model.")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation MCC: {best_mcc}")
    return best_model_path
