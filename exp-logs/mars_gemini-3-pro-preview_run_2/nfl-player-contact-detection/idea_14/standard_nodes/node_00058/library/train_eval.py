import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from tqdm import (
    tqdm,
)  # Not strictly required by prompt but useful, will suppress if needed or just use simple print

from library.config import Config
from library.loss import FocalLoss
from library.model import ECGRN
from library.dataset import get_dataloaders
from library.data_processing import DataProcessor


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """Calculates Matthews Correlation Coefficient."""
    # Ensure inputs are numpy arrays/integers
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_probs):
    """
    Performs a grid search to find the best threshold for MCC.

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1).
        y_probs (np.ndarray): Predicted probabilities (0 to 1).

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Search range: 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, 0.01)

    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)
        mcc = compute_mcc(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for X_cont, X_cat, y in loader:
        X_cont = X_cont.to(device)
        X_cat = X_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass (returns logits)
        logits = model(X_cont, X_cat)

        # Loss calculation
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        count += y.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Runs validation and collects predictions.

    Returns:
        tuple: (avg_loss, y_true_np, y_probs_np)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_logits = []

    with torch.no_grad():
        for X_cont, X_cat, y in loader:
            X_cont = X_cont.to(device)
            X_cat = X_cat.to(device)
            y = y.to(device)

            logits = model(X_cont, X_cat)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate and convert
    logits_tensor = torch.cat(all_logits)
    probs_tensor = torch.sigmoid(logits_tensor)

    y_true_np = torch.cat(all_targets).numpy().flatten()
    y_probs_np = probs_tensor.numpy().flatten()

    return avg_loss, y_true_np, y_probs_np


def train_model(config: Config, processor: DataProcessor):
    """
    Main function to train the ECGRN model.

    Args:
        config (Config): Configuration object.
        processor (DataProcessor): Data processor instance.

    Returns:
        tuple: (best_model, best_threshold)
    """
    set_seed(config.SEED)

    # 1. Get Dataloaders
    print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(config, processor)

    # 2. Determine Model Input Dimensions
    # Continuous dims: Check the shape of a batch from the dataset
    # The dataset is already loaded in memory inside the loader's dataset
    num_continuous = train_loader.dataset.X_cont.shape[1]

    # Categorical dims: Based on config and processing logic
    # Order in DataProcessor: ["position_1", "team_1", "position_2", "team_2"]
    # Config Dims: "position", "team"
    cat_embedding_dims = [
        config.EMBEDDING_DIMS["position"],  # position_1
        config.EMBEDDING_DIMS["team"],  # team_1
        config.EMBEDDING_DIMS["position"],  # position_2
        config.EMBEDDING_DIMS["team"],  # team_2
    ]

    print(
        f"Model Input: Continuous={num_continuous}, Categorical Embeddings={cat_embedding_dims}"
    )

    # 3. Initialize Model, Loss, Optimizer
    model = ECGRN(
        num_continuous=num_continuous,
        categorical_embedding_dims=cat_embedding_dims,
        hidden_size=config.HIDDEN_SIZE,
        num_blocks=config.NUM_BLOCKS,
        dropout_rate=config.DROPOUT,
    ).to(config.DEVICE)

    criterion = FocalLoss(
        alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, reduction="mean"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_val_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {config.EPOCHS} epochs on {config.DEVICE}...")

    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )

        # Validate
        val_loss, y_true, y_probs = validate(
            model, val_loader, criterion, config.DEVICE
        )

        # Optimize Threshold
        curr_thresh, curr_mcc = optimize_threshold(y_true, y_probs)

        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCC: {curr_mcc:.8f} | "
            f"Best Thresh: {curr_thresh:.2f}"
        )

        # Early Stopping Logic
        if curr_mcc > best_val_mcc:
            best_val_mcc = curr_mcc
            best_threshold = curr_thresh
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! MCC: {best_val_mcc:.8f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Load Best Model
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

    print(
        f"Training complete. Best Validation MCC: {best_val_mcc:.8f} at Threshold: {best_threshold:.2f}"
    )

    return model, best_threshold
