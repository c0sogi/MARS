import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint, load_checkpoint
from library.model import HybridResFunnel


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer instance.
        criterion: The loss function.
        device: The device to run on ('cpu' or 'cuda').

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Unpack data
        cont_data = batch["continuous"].to(device, non_blocking=True)
        cat_data = batch["categorical"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(cont_data, cat_data)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cont_data = batch["continuous"].to(device, non_blocking=True)
            cat_data = batch["categorical"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)

            outputs = model(cont_data, cat_data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            num_batches += 1

            # Store for AUC calculation
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate and compute AUC
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0

    return avg_loss, auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for test data.
        device: The device to run on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cont_data = batch["continuous"].to(device, non_blocking=True)
            cat_data = batch["categorical"].to(device, non_blocking=True)

            outputs = model(cont_data, cat_data)
            all_preds.append(outputs.cpu().numpy())

    if len(all_preds) > 0:
        return np.concatenate(all_preds).flatten()
    else:
        return np.array([])


def train_model(
    train_loader,
    val_loader,
    test_loader,
    test_ids,
    epochs=Config.EPOCHS,
    patience=5,
    device=Config.DEVICE,
):
    """
    Orchestrates the training process, including early stopping and submission generation.

    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for testing.
        test_ids: Array of test IDs.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        device: Device to run on.
    """
    print(f"Starting training on device: {device}")

    # Initialize Model
    model = HybridResFunnel().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # Loss Function (Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Tracking
    best_auc = 0.0
    patience_counter = 0
    checkpoint_dir = Config.WORKING_DIR

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc": best_auc,
            },
            is_best,
            checkpoint_dir,
        )

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {epoch} epochs. Best Val AUC: {best_auc}"
            )
            break

    # --- Submission Generation ---
    print("Generating submission...")

    # Load best model
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model not found. Using current model state.")

    # Predict
    predictions = predict(model, test_loader, device)

    # Save to CSV
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
