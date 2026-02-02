import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from library.config import Config
from library.model import VNCGCNN
from library.data import get_dataloaders


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Compute loss (MSE)
        loss = criterion(outputs, batch.y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)
            loss = criterion(outputs, batch.y)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def generate_submission(model, loader, target_scaler, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids = []
    predictions = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            outputs = batch_outputs = model(batch)

            # Inverse transform predictions to original scale
            # outputs is (B, 2), scaler expects numpy (B, 2)
            outputs_np = outputs.cpu().numpy()
            if target_scaler is not None:
                outputs_np = target_scaler.inverse_transform(outputs_np)

            # Collect IDs and predictions
            batch_ids = batch.id.cpu().numpy().flatten()

            ids.extend(batch_ids)
            predictions.append(outputs_np)

    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to match sample submission structure usually
    df = df.sort_values("id")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    # 1. Set Seed
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Get DataLoaders
    # This handles caching of graph data internally
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Initialize Model
    model = VNCGCNN(Config).to(device)

    # 4. Optimizer and Scheduler
    # Using AdamW as specified in the idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    criterion = nn.MSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # 6. Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(
        model, test_loader, target_scaler, device, Config.SUBMISSION_PATH
    )
