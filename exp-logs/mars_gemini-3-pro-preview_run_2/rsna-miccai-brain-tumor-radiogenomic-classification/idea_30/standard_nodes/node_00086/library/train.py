import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.model import AsymmetricEfficientNet
from library.data import get_dataloaders

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKING_DIR = "./working/idea_opt"
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Training & Validation Steps
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The computing device.

    Returns:
        avg_loss: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The computing device.

    Returns:
        avg_loss: Average validation loss.
        auc_score: ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Handle case where only one class is present in batch
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_preds)
        else:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return avg_loss, auc_score


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def run_training(
    max_epochs=20,
    patience=5,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-2,
    num_workers=4,
    seed=42,
    load_cached_roi=True,
):
    """
    Orchestrates the training process.

    Args:
        max_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for AdamW.
        weight_decay (float): Weight decay for AdamW.
        num_workers (int): Number of dataloader workers.
        seed (int): Random seed.
        load_cached_roi (bool): Whether to use cached ROI data.
    """
    set_seed(seed)
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Starting training on device: {DEVICE}")

    # 1. Data Loading
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, load_cached_roi=load_cached_roi
    )

    # 2. Model Initialization
    model = AsymmetricEfficientNet(pretrained=True, dropout_rate=0.5)
    model = model.to(DEVICE)

    # 3. Optimization Setup
    # Using BCEWithLogitsLoss as model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()

    # AdamW with specified parameters
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    best_auc = 0.0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping & Model Checkpointing
        # We monitor Validation Loss for early stopping as per standard practice,
        # though we also track AUC.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved to {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation Loss: {best_val_loss}")
    print(f"Best Validation AUC: {best_auc}")

    return model
