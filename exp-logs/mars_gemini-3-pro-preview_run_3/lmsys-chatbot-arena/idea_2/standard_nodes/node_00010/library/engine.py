import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    print_metrics,
)
from library.data import prepare_data
from library.model import ESIMHybridModel


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move batch to device
        prompt_ids = batch["prompt_ids"].to(device)
        res_a_ids = batch["res_a_ids"].to(device)
        res_b_ids = batch["res_b_ids"].to(device)
        scalars = batch["scalars"].to(device)
        targets = batch["target"].to(device)

        batch_size = prompt_ids.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(prompt_ids, res_a_ids, res_b_ids, scalars)

        # Compute loss (CrossEntropyLoss supports soft targets)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in dataloader:
            prompt_ids = batch["prompt_ids"].to(device)
            res_a_ids = batch["res_a_ids"].to(device)
            res_b_ids = batch["res_b_ids"].to(device)
            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            batch_size = prompt_ids.size(0)

            logits = model(prompt_ids, res_a_ids, res_b_ids, scalars)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns a numpy array of probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            prompt_ids = batch["prompt_ids"].to(device)
            res_a_ids = batch["res_a_ids"].to(device)
            res_b_ids = batch["res_b_ids"].to(device)
            scalars = batch["scalars"].to(device)

            logits = model(prompt_ids, res_a_ids, res_b_ids, scalars)

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    return np.vstack(preds)


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main training loop with early stopping.
    """
    seed_everything()
    Config.setup()

    # Load Data
    # We rely on prepare_data's internal caching mechanism
    train_loader, val_loader, _, _ = prepare_data(
        load_cached_data=True, batch_size=batch_size, debug=debug
    )

    # Initialize Model
    model = ESIMHybridModel().to(Config.DEVICE)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)

        print(f"Epoch {epoch + 1}/{epochs}")
        print_metrics({"Train Loss": train_loss, "Val Loss": val_loss})

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def generate_submission(model_path):
    """
    Loads the best model, predicts on the test set, and saves the submission file.
    """
    seed_everything()
    Config.setup()

    # Load Test Data
    _, _, test_loader, _ = prepare_data(load_cached_data=True)

    # Load Model
    model = ESIMHybridModel().to(Config.DEVICE)
    state_dict = load_checkpoint(model_path, Config.DEVICE)

    if state_dict is None:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(state_dict)
    print(f"Loaded model from {model_path}")

    # Predict
    print("Generating predictions...")
    probs = predict(model, test_loader, Config.DEVICE)

    # Load Test Metadata to get IDs
    # We read this directly as it's a simple CSV read
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Ensure alignment
    if len(test_df) != len(probs):
        raise ValueError(
            f"Mismatch: Test set has {len(test_df)} rows, but generated {len(probs)} predictions."
        )

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": probs[:, 0],
            "winner_model_b": probs[:, 1],
            "winner_tie": probs[:, 2],
        }
    )

    # Save
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
