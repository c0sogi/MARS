import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config, seed_everything
from library.model import SiameseLSTM
from library.dataset import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cpu or cuda).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move data to device
        prompt = batch["prompt"].to(device)
        res_a = batch["response_a"].to(device)
        res_b = batch["response_b"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(prompt, res_a, res_b, lengths)

        # Compute loss
        # CrossEntropyLoss supports soft targets (probabilities) directly
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            prompt = batch["prompt"].to(device)
            res_a = batch["response_a"].to(device)
            res_b = batch["response_b"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["target"].to(device)

            logits = model(prompt, res_a, res_b, lengths)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        dataloader: Test DataLoader.
        device: Device to run on.

    Returns:
        np.ndarray: Array of shape (N, 3) containing probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            prompt = batch["prompt"].to(device)
            res_a = batch["response_a"].to(device)
            res_b = batch["response_b"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(prompt, res_a, res_b, lengths)

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        return np.concatenate(all_probs, axis=0)
    return np.array([])


def run_training(config: Config):
    """
    Main training loop with Early Stopping.

    Args:
        config: Configuration object.
    """
    seed_everything(config.SEED)

    # 1. Prepare Data
    print("Preparing data...")
    train_loader, val_loader, _, _ = get_dataloaders(config, load_cached_data=True)

    # 2. Initialize Model
    print("Initializing model...")
    model = SiameseLSTM(config)
    model.to(config.DEVICE)

    # 3. Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {config.DEVICE}...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss = validate(model, val_loader, criterion, config.DEVICE)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"Validation loss improved. Model saved to {config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")


def generate_submission(config: Config):
    """
    Generates the submission file using the best trained model.

    Args:
        config: Configuration object.
    """
    seed_everything(config.SEED)

    # 1. Load Data
    # We need the test loader for predictions and the raw test csv for IDs
    _, _, test_loader, _ = get_dataloaders(config, load_cached_data=True)

    # Load test metadata to get IDs
    if not os.path.exists(config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {config.TEST_DATA_PATH}")

    test_df = pd.read_csv(config.TEST_DATA_PATH)

    # 2. Load Model
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Run training first."
        )

    print(f"Loading model from {config.MODEL_PATH}...")
    model = SiameseLSTM(config)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)

    # 3. Predict
    print("Generating predictions...")
    probs = predict(model, test_loader, config.DEVICE)

    # 4. Format Submission
    if len(probs) != len(test_df):
        raise ValueError(
            f"Mismatch in prediction count: {len(probs)} vs {len(test_df)}"
        )

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": probs[:, 0],
            "winner_model_b": probs[:, 1],
            "winner_tie": probs[:, 2],
        }
    )

    # 5. Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

    # Print head for verification
    print("Submission Head:")
    print(submission_df.head())
