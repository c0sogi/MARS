import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.model import InputInjectedFunnelMLP
from library.data_utils import get_data_loaders


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: Calculation device (CPU/GPU).
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for x_cont, x_cat, y in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x_cont, x_cat)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: Calculation device.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            logits = model(x_cont, x_cat)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            running_loss += loss.item()
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / len(loader)
    auc_score = roc_auc_score(all_targets, all_preds)

    return avg_loss, auc_score


def predict(model, loader, device):
    """
    Generates probability predictions for the test set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for test data.
        device: Calculation device.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat in loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug_limit=None,
):
    """
    Orchestrates the full training pipeline:
    1. Loads data.
    2. Initializes model, optimizer, and scheduler.
    3. Runs training loop with Early Stopping.
    4. Saves the best model.
    5. Generates submission file.

    Args:
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to use cached preprocessed data.
        debug_limit (int, optional): Limit training samples for debugging.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader, vocab_sizes, cont_dim = get_data_loaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 2. Initialize Model
    model = InputInjectedFunnelMLP(
        cont_dim=cont_dim,
        vocab_sizes=vocab_sizes,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT,
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_auc}")

    # 5. Inference & Submission
    if os.path.exists(best_model_path):
        print("Loading best model for inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions on test set...")
    test_preds = predict(model, test_loader, device)

    # Load test IDs from metadata to ensure alignment
    test_df = pd.read_csv(Config.TEST_PATH)
    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
