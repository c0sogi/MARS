import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time

from library.config import Config
from library.utils import set_seed, TargetScaler
from library.data import get_dataloaders
from library.model import OptimizedCGCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
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
        targets = batch.y

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on a given dataset (validation).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)
            targets = batch.y

            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    print("Generating submission...")
    model.eval()

    # Load the scaler to inverse transform predictions
    scaler = TargetScaler()
    if os.path.exists(Config.TARGET_SCALER_CACHE):
        scaler.load(Config.TARGET_SCALER_CACHE)
    else:
        print(
            "Warning: Target scaler cache not found. Predictions will be on standardized scale."
        )

    ids = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Predict
            outputs = model(batch)

            # Inverse transform to original scale
            if scaler.mean is not None:
                outputs = scaler.inverse_transform(outputs)

            preds_list.append(outputs.cpu().numpy())
            ids.extend(batch.id.cpu().numpy())

    if len(preds_list) > 0:
        preds_array = np.concatenate(preds_list, axis=0)

        # Create DataFrame
        df = pd.DataFrame(preds_array, columns=Config.TARGET_COLS)
        df.insert(0, "id", ids)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print("No predictions generated.")


def run_training(load_cached_data=True, num_epochs=Config.NUM_EPOCHS):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = OptimizedCGCNN(config=Config).to(device)

    # 3. Setup Optimizer, Loss, Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Using MSE Loss on standardized targets
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"New best model saved with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # 5. Generate Submission using the best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    return best_val_loss
