import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_lrap, save_checkpoint, set_seed


def mixup_data(x, y, alpha=0.4):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, mixed targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # For multi-label BCE, we can mix the targets directly
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y, lam


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch with Mixup augmentation.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Apply Mixup with probability 0.5 (Cite solution_lesson_node_00004)
        if np.random.random() < 0.5:
            inputs, targets, _ = mixup_data(inputs, targets, alpha=0.4)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Validation LRAP score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)

            # Accumulate loss
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(logits)

            all_outputs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate LRAP
    lrap = calculate_lrap(all_targets, all_outputs)

    return avg_loss, lrap


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=Config.MAX_EPOCHS,
    lr=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    save_path=Config.MODEL_PATH,
):
    """
    Orchestrates the training process with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Compute device.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_lrap = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val LRAP: {val_lrap}")

        # Early Stopping and Checkpointing based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filepath=save_path,
            )
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def predict(model, test_loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Compute device.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    all_probs = []

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Convert to probabilities
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all predictions
    all_probs = np.concatenate(all_probs, axis=0)

    # Retrieve filenames from the dataset
    # Note: We assume the loader preserves order and is not shuffled
    fnames = test_loader.dataset.df["fname"].values

    # Retrieve class names (columns 1 to 80 in sample submission)
    # We can get this from the test dataframe columns excluding metadata
    # The dataset object has 'label_cols' which are the class names
    class_names = test_loader.dataset.label_cols

    # Create Submission DataFrame
    submission_df = pd.DataFrame(all_probs, columns=class_names)
    submission_df.insert(0, "fname", fnames)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
