import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import SSVEDataset
from library.model import SSVEModel


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Stochastic View Selection.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        # images shape: (Batch, 64, 256, 256) - Randomly View A or View B
        # targets shape: (Batch,)

        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape: (Batch, 1)

        optimizer.zero_grad()

        outputs = model(images)  # Logits
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model using Multi-View Ensemble (View A + View B).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            # images shape: (Batch, 2, 64, 256, 256) - Both views
            # targets shape: (Batch,)

            targets_gpu = targets.to(device).unsqueeze(1)

            # Split views
            view_a = images[:, 0, ...].to(device)  # (Batch, 64, 256, 256)
            view_b = images[:, 1, ...].to(device)  # (Batch, 64, 256, 256)

            # Forward pass for both views
            logits_a = model(view_a)
            logits_b = model(view_b)

            # Calculate loss (optional, using average logits or one of them,
            # but usually we care about AUC here. We'll average logits for loss calculation stability)
            avg_logits = (logits_a + logits_b) / 2.0
            loss = criterion(avg_logits, targets_gpu)
            running_loss += loss.item() * images.size(0)

            # Compute probabilities
            probs_a = torch.sigmoid(logits_a)
            probs_b = torch.sigmoid(logits_b)

            # Ensemble Average
            avg_probs = (probs_a + probs_b) / 2.0

            all_targets.extend(targets.numpy())
            all_preds.extend(avg_probs.cpu().numpy().flatten())

    val_loss = running_loss / len(loader.dataset)

    # Calculate AUC
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5  # Handle edge case with single class in batch

    if np.isnan(val_auc):
        val_auc = 0.5

    return val_loss, val_auc


def run_training(load_cached_data=True):
    """
    Main execution function for the training pipeline.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Initializing Datasets...")
    train_dataset = SSVEDataset(mode="train", load_cached_data=load_cached_data)
    val_dataset = SSVEDataset(mode="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    print("Initializing Model...")
    model = SSVEModel()
    model.to(device)

    # 3. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0

    print("Starting Training...")
    print("-" * 60)

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"--> Best AUC improved. Model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("-" * 60)
    print(f"Training Complete. Best Validation AUC: {best_auc}")
