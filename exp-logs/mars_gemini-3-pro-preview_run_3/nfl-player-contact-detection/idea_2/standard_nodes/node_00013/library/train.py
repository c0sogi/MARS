import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

from library.config import Config
from library.utils import set_seed, compute_mcc, optimize_threshold
from library.dataset import get_dataloader
from library.model import TemporalCNN


def train_epoch(model, dataloader, criterion, optimizer, device, debug=False):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if debug and i >= 10:
            break

    return running_loss / num_batches


def validate(model, dataloader, criterion, device, debug=False):
    """
    Evaluates the model on the validation set.
    Returns average loss, true labels, and predicted probabilities.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            all_labels.append(labels.cpu().numpy())
            all_probs.append(outputs.cpu().numpy())

            if debug and i >= 10:
                break

    avg_loss = running_loss / num_batches
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)

    return avg_loss, y_true, y_probs


def train_model(
    num_epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=3,
    load_cached_data=True,
    debug=False,
):
    """
    Main function to train the model, handle early stopping, and optimize threshold.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = TemporalCNN().to(device)

    # Optimizer and Loss
    # Using BCELoss because the model output includes a Sigmoid activation
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # DataLoaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        "train", batch_size=batch_size, load_cached_data=load_cached_data
    )
    val_loader = get_dataloader(
        "validation", batch_size=batch_size, load_cached_data=load_cached_data
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, debug=debug
        )
        val_loss, val_true, val_probs = validate(
            model, val_loader, criterion, device, debug=debug
        )

        # Compute MCC with default threshold 0.5 for monitoring
        val_preds_default = (val_probs >= 0.5).astype(int)
        val_mcc = compute_mcc(val_true, val_preds_default)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MCC (0.5): {val_mcc:.6f}"
        )

        # Early Stopping based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Ensure directory exists
            os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        if debug:
            print("Debug mode active, stopping after 1 epoch.")
            break

    # Load best model for threshold optimization
    print("Loading best model for threshold optimization...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Optimize Threshold
    print("Optimizing threshold on validation set...")
    _, val_true, val_probs = validate(model, val_loader, criterion, device, debug=debug)
    best_threshold = optimize_threshold(val_true, val_probs)

    # Recalculate best MCC
    best_preds = (val_probs >= best_threshold).astype(int)
    best_mcc = compute_mcc(val_true, best_preds)
    print(
        f"Final Best Validation MCC: {best_mcc:.6f} at Threshold: {best_threshold:.2f}"
    )

    return model, best_threshold


def generate_submission(
    model, threshold, batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=False
):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Generating submission...")
    test_loader = get_dataloader(
        "test", batch_size=batch_size, load_cached_data=load_cached_data
    )

    all_probs = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_probs.append(outputs.cpu().numpy())

            if debug and i >= 10:
                break

    y_probs = np.concatenate(all_probs)

    # Apply threshold
    y_preds = (y_probs >= threshold).astype(int)

    # Get IDs
    # If debug was active, we slice the IDs to match the number of predictions
    if debug:
        ids = test_loader.dataset.ids[: len(y_preds)]
    else:
        ids = test_loader.dataset.ids

    # Flatten predictions (y_preds is (N, 1))
    y_preds = y_preds.flatten()

    # Create DataFrame
    df_sub = pd.DataFrame({"contact_id": ids, "contact": y_preds})

    # Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return df_sub
