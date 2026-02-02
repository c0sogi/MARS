import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import HybridTransformer


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for x_seq, x_num, targets in loader:
        x_seq = x_seq.to(device)
        x_num = x_num.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x_seq, x_num)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * x_seq.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_seq, x_num, targets in loader:
            x_seq = x_seq.to(device)
            x_num = x_num.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(x_seq, x_num)
            loss = criterion(logits, targets)

            running_loss += loss.item() * x_seq.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(train_loader, val_loader, epochs=Config.EPOCHS, patience=5):
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = HybridTransformer().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0

    # Ensure directory for saving model exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    print(
        f"Starting training on {device} for {epochs} epochs with patience {patience}..."
    )

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved (AUC: {best_val_auc})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training Complete. Best Validation AUC: {best_val_auc}")
    return best_val_auc


def predict(test_loader):
    """
    Generates predictions using the best saved model and creates a submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    model = HybridTransformer().to(device)

    # Load Best Model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    model.eval()
    all_preds = []

    # Generate Probabilities
    with torch.no_grad():
        for x_seq, x_num in test_loader:
            x_seq = x_seq.to(device)
            x_num = x_num.to(device)

            logits = model(x_seq, x_num)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    predictions = np.concatenate(all_preds).flatten()

    # Load Test IDs to ensure alignment
    if os.path.exists(Config.TEST_PATH):
        df_test = pd.read_csv(Config.TEST_PATH)
        ids = df_test["id"].values
    else:
        raise FileNotFoundError(f"Test metadata file not found at {Config.TEST_PATH}")

    # Handle potential size mismatch if debug sampling was used
    if len(predictions) != len(ids):
        if len(predictions) < len(ids):
            print(
                f"Warning: Prediction count ({len(predictions)}) is less than ID count ({len(ids)}). "
                "Assuming debug mode and slicing IDs."
            )
            ids = ids[: len(predictions)]
        else:
            raise ValueError(
                f"Mismatch: {len(ids)} IDs vs {len(predictions)} predictions"
            )

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": ids, "target": predictions})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
