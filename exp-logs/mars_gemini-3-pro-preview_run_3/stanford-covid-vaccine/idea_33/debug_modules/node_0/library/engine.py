import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import MCRMSELoss, metric_calculator


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Device (cuda/cpu).
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    criterion = MCRMSELoss()

    for batch in loader:
        inputs = batch["inputs"].to(device)
        adj_map = batch["adj_map"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, adj_map)

        # Loss calculation (MCRMSE on all 5 targets)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the official competition metric.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: Device (cuda/cpu).

    Returns:
        float: The calculated MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            adj_map = batch["adj_map"].to(device)
            targets = batch["targets"]  # Keep targets on CPU for aggregation

            # Forward pass
            outputs = model(inputs, adj_map)

            # Move outputs to CPU
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Global Aggregation: Concatenate all batches
    # Preds shape: (N, 107, 5)
    # Targets shape: (N, 68, 5)
    global_preds = np.concatenate(all_preds, axis=0)
    global_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    # The utility function handles slicing to seq_scored (68) and filtering columns
    score = metric_calculator(global_preds, global_targets)

    return score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.

    Returns:
        float: Best validation score achieved.
    """
    best_score = float("inf")
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Logging (Full precision for validation score)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best MCRMSE: {best_score}")
    return best_score


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: PyTorch model (should be loaded with best weights).
        test_loader: DataLoader for test data.
        device: Device.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            adj_map = batch["adj_map"].to(device)
            ids = batch["id"]

            outputs = model(inputs, adj_map)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    preds = np.concatenate(all_preds, axis=0)

    # Flatten logic for submission format
    # Format requires one row per sequence position: id_{id}_{pos}
    n_samples, seq_len, n_targets = preds.shape
    target_cols = Config.TARGET_COLS

    flat_ids = []

    # Generate ID column
    for i in range(n_samples):
        sample_id = all_ids[i]
        for j in range(seq_len):
            flat_ids.append(f"{sample_id}_{j}")

    # Flatten predictions to (N_samples * 107, 5)
    flat_preds = preds.reshape(-1, n_targets)

    # Create DataFrame
    df = pd.DataFrame(flat_preds, columns=target_cols)
    df.insert(0, "id_seqpos", flat_ids)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
