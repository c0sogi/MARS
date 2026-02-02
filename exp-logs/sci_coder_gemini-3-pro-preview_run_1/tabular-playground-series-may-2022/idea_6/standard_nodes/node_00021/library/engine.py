import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model to train.
        loader: DataLoader for the training set.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        criterion: The loss function.
        device: The device to run on (cpu or cuda).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_seq = batch["x_seq"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_num, x_seq)
        loss = criterion(logits, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader for the validation set.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)
            target = batch["target"].to(device)

            logits = model(x_num, x_seq)
            loss = criterion(logits, target)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0

    return avg_loss, auc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    config: Config,
):
    """
    Manages the full training loop with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        criterion: The loss function.
        device: The device to run on.
        config: Configuration object containing hyperparameters.

    Returns:
        float: The best validation AUC achieved.
    """
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc + config.early_stopping_min_delta:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(model, test_loader, test_ids, device, config: Config):
    """
    Generates predictions for the test set and saves to submission.csv.

    Args:
        model: The PyTorch model.
        test_loader: DataLoader for the test set.
        test_ids: Numpy array of test IDs.
        device: The device to run on.
        config: Configuration object.
    """
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    # Load best model if available
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)

            logits = model(x_num, x_seq)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds).flatten()
    else:
        all_preds = np.zeros(len(test_ids))

    # Ensure lengths match
    if len(all_preds) != len(test_ids):
        print(
            f"Warning: Prediction count {len(all_preds)} does not match ID count {len(test_ids)}"
        )

    submission = pd.DataFrame({config.id_col: test_ids, config.target_col: all_preds})

    print(f"Saving submission to {config.submission_path}...")
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    submission.to_csv(config.submission_path, index=False)
    print("Submission saved successfully.")
