import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.dataset import get_dataloaders
from library.model import FrozenResNetLinear


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for images, metadata, targets in dataloader:
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images, metadata)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    return running_loss / num_samples


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and RMSE.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, metadata, targets in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, metadata)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    epoch_loss = running_loss / num_samples

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)

    return epoch_loss, rmse


def inference(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, metadata, _ in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)
            preds = outputs.cpu().numpy().flatten()
            predictions.extend(preds)

    return np.array(predictions)


def train_model(debug=False):
    """
    Main training loop with Early Stopping.
    """
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device(Config.DEVICE)

    # Get Dataloaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Initialize Model
    model = FrozenResNetLinear().to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    # Optimize only the head parameters
    optimizer = optim.AdamW(model.head.parameters(), lr=Config.LEARNING_RATE)

    # Training Configuration
    num_epochs = Config.NUM_EPOCHS
    best_rmse = float("inf")
    patience = 5
    patience_counter = 0

    # Ensure model directory exists
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val RMSE: {val_rmse} - "
            f"Time: {elapsed}s"
        )

        # Early Stopping and Model Saving
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_rmse


def predict_and_submit(debug=False):
    """
    Loads the best model, generates predictions on the test set, and saves to CSV.
    """
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(debug=debug)

    # Load Model
    model = FrozenResNetLinear().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Generate predictions
    raw_predictions = inference(model, test_loader, device)

    # Clip predictions to valid range [1, 100]
    predictions = np.clip(raw_predictions, 1.0, 100.0)

    # Retrieve Ids from the dataset dataframe
    test_df = test_loader.dataset.df
    ids = test_df["Id"].values

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
