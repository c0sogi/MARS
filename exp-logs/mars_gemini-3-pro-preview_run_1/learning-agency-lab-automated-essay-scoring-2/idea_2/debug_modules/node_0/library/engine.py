import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import compute_qwk


def train_one_epoch(model, optimizer, scheduler, dataloader, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloader: The training DataLoader.
        device: The device to run training on.
        criterion: The loss function.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for data in dataloader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss (flatten outputs and targets to ensure shape match)
        loss = criterion(outputs.view(-1), targets.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        # Update statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation DataLoader.
        device: The device to run evaluation on.
        criterion: The loss function.

    Returns:
        tuple: (average_loss, qwk_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["labels"].to(device)

            batch_size = input_ids.size(0)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Calculate loss
            loss = criterion(outputs.view(-1), targets.view(-1))

            # Update statistics
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions and targets
            batch_preds = outputs.view(-1).detach().cpu().numpy()
            batch_labels = targets.view(-1).detach().cpu().numpy()

            preds.extend(batch_preds)
            labels.extend(batch_labels)

    val_loss = running_loss / dataset_size

    # Compute Quadratic Weighted Kappa
    val_qwk = compute_qwk(labels, preds)

    return val_loss, val_qwk


def predict(model, dataloader, device):
    """
    Generates raw predictions for a dataset.

    Args:
        model: The PyTorch model.
        dataloader: The DataLoader (test set).
        device: The device to run inference on.

    Returns:
        list: List of predicted scores (floats).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            batch_preds = outputs.view(-1).detach().cpu().numpy()
            preds.extend(batch_preds)

    return preds


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience=3
):
    """
    Main training loop with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Scheduler.
        device: Device.
        epochs: Number of epochs.
        patience: Patience for early stopping.

    Returns:
        model: The model loaded with the best weights.
    """
    criterion = nn.MSELoss()
    best_qwk = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, criterion
        )
        val_loss, val_qwk = validate(model, val_loader, device, criterion)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_qwk}")

        # Early Stopping Logic
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save the best model
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"Validation QWK improved. Model saved to {Config.model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in QWK. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def generate_submission(model, test_loader, device, output_path=Config.submission_path):
    """
    Generates the submission file for the test set.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Device.
        output_path: Path to save the submission CSV.
    """
    print("Generating submission...")

    # Get raw predictions
    raw_preds = predict(model, test_loader, device)

    # Post-process predictions
    # Clip to [1, 6] and round to nearest integer
    final_preds = np.array(raw_preds)
    final_preds = np.clip(final_preds, 1, 6)
    final_preds = np.round(final_preds).astype(int)

    # Retrieve essay_ids from the dataset
    # Note: We assume the test_loader is not shuffled (shuffle=False in dataset.py)
    essay_ids = test_loader.dataset.df["essay_id"].values

    # Create DataFrame
    submission_df = pd.DataFrame({"essay_id": essay_ids, "score": final_preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
