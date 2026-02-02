import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import pandas as pd

from library.config import Config
from library.data_loader import get_loaders
from library.model import DC3_WDS


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move batch to device
        atomic_feats = batch["atomic_feats"].to(device)
        global_feats = batch["global_feats"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_feats, global_feats, mask)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            atomic_feats = batch["atomic_feats"].to(device)
            global_feats = batch["global_feats"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic_feats, global_feats, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

    total_loss = running_loss / len(dataloader.dataset)
    return total_loss


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    predictions = []
    ids = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch in test_loader:
            atomic_feats = batch["atomic_feats"].to(device)
            global_feats = batch["global_feats"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"].cpu().numpy()

            # Forward pass
            outputs = model(atomic_feats, global_feats, mask)

            # Inverse transform: exp(x) - 1
            # The model predicts log1p(target), so we apply expm1 to get back to original scale
            preds = torch.expm1(outputs).cpu().numpy()

            predictions.append(preds)
            ids.append(batch_ids)

    predictions = np.concatenate(predictions, axis=0)
    ids = np.concatenate(ids, axis=0)

    # Create DataFrame
    df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
    df.insert(0, "id", ids)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True):
    """
    Main function to run the training process.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = DC3_WDS().to(device)

    # 3. Define Loss and Optimizer
    # MSE Loss on log-transformed targets corresponds to squared RMSLE
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics (Full precision as requested)
        # Note: val_loss is MSE of log-transformed targets.
        # RMSLE approx = sqrt(val_loss)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss (MSE): {train_loss} | Val Loss (MSE): {val_loss} | Val RMSLE: {np.sqrt(val_loss)}"
        )

        # Early Stopping and Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Val Loss: {best_val_loss}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
            )
            break

    print(f"Training completed. Best Val Loss: {best_val_loss}")

    # 5. Generate Submission
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
