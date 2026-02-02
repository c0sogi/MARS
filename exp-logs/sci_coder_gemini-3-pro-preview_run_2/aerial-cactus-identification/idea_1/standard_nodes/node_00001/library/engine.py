import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint, load_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        criterion (torch.nn.modules.loss._Loss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on (CPU or GPU).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.modules.loss._Loss): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.array([])
        all_labels = np.array([])

    # Calculate ROC AUC
    # Handle potential edge cases where a batch might only have one class
    try:
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_preds)
        else:
            auc_score = 0.5
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def train_model(model, train_loader, val_loader, config=Config):
    """
    Main training loop with Early Stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        config (class): Configuration class with hyperparameters.

    Returns:
        torch.nn.Module: The trained model with best weights loaded.
    """
    device = torch.device(config.DEVICE)
    model = model.to(device)

    # Binary Cross Entropy with Logits is more numerically stable
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_auc = 0.0
    patience_counter = 0

    # Ensure model save directory exists
    if config.MODEL_SAVE_PATH:
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print(f"Starting training on device: {device}")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc + config.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0

            # Save the best model state
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                config.MODEL_SAVE_PATH,
            )

        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load the best model weights before returning
    if os.path.exists(config.MODEL_SAVE_PATH):
        load_checkpoint(config.MODEL_SAVE_PATH, model)

    return model


def predict_and_submit(model, test_loader, output_path, device):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        output_path (str): Path to save the submission CSV.
        device (torch.device): Device to run inference on.
    """
    model.eval()
    model = model.to(device)

    probs = []

    # Ensure no gradients are calculated
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Apply sigmoid to convert logits to probabilities
            batch_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            probs.extend(batch_probs)

    # Retrieve IDs from the dataset metadata
    # We assume the test_loader is sequential (shuffle=False)
    test_metadata = test_loader.dataset.metadata
    ids = test_metadata["id"].values

    if len(ids) != len(probs):
        print(
            f"Warning: Mismatch between number of IDs ({len(ids)}) and predictions ({len(probs)})"
        )

    # Create submission DataFrame
    df = pd.DataFrame({"id": ids, "has_cactus": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
