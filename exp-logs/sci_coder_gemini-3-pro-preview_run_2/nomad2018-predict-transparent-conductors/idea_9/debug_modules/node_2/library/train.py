import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.data import get_dataloaders
from library.model import DualStreamCGCNN


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, batch.y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            outputs = model(batch)
            loss = criterion(outputs, batch.y)

            running_loss += loss.item() * batch.num_graphs

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


def run_training(load_cached_data=True):
    """
    Main function to manage the training process.

    Args:
        load_cached_data (bool): Whether to try loading pre-processed data from cache.

    Returns:
        model: The trained PyTorch model.
        target_scaler: The fitted scaler for target variables.
        test_loader: DataLoader for the test set.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load data
    print("Preparing data...")
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize model
    device = torch.device(Config.DEVICE)
    model = DualStreamCGCNN().to(device)

    # Define optimizer and loss
    # Using AdamW as specified in the plan
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Mean Squared Error Loss for standardized targets
    criterion = nn.MSELoss()

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{Config.MAX_EPOCHS} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss: {val_loss:.10f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print(f"  New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.10f}")

    # Load best model for inference
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    return model, target_scaler, test_loader


def generate_submission(model, test_loader, target_scaler):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        target_scaler: Fitted StandardScaler to inverse transform predictions.
    """
    print("Generating submission...")
    device = torch.device(Config.DEVICE)
    model.eval()

    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass (outputs are standardized)
            outputs_scaled = model(batch)

            # Inverse transform to get original units (eV)
            outputs_original = target_scaler.inverse_transform(outputs_scaled)

            # Collect results
            ids.extend(batch.id.cpu().numpy().flatten())
            predictions.append(outputs_original.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to ensure correct order (though not strictly required by CSV format, it's good practice)
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
