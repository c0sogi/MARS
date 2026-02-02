import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library import config
from library.dataset import get_dataloader
from library.model import HCPDS


def set_seed(seed=config.SEED):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        # Move data to device
        atomic_feat = batch["atomic_features"].to(device)
        global_feat = batch["global_features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(atomic_feat, global_feat, mask)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss (MSE is averaged over batch, so multiply by batch size)
        running_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average MSE loss (which corresponds to MSLE on original scale).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            atomic_feat = batch["atomic_features"].to(device)
            global_feat = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic_feat, global_feat, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    lr=config.LEARNING_RATE,
    patience=config.PATIENCE,
    load_cached_data=True,
):
    """
    Main training loop with early stopping and scheduler.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # Load Data
    train_loader = get_dataloader(
        "train", batch_size=batch_size, shuffle=True, load_cached_data=load_cached_data
    )
    val_loader = get_dataloader(
        "val", batch_size=batch_size, shuffle=False, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = HCPDS().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    # Loss Function: MSE on log-transformed targets
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # RMSLE is sqrt of MSLE (which is our val_loss since targets are logs)
        val_rmsle = np.sqrt(val_loss)

        # Update Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train MSE (Log): {train_loss:.10f} | Val MSE (Log): {val_loss:.10f} | Val RMSLE: {val_rmsle:.10f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Ensure directory exists
            os.makedirs(os.path.dirname(config.MODEL_CHECKPOINT), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT)
            print(f"  -> New best model saved with Val Loss: {best_val_loss:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Loss (MSE on Log): {best_val_loss:.10f}")


def generate_submission(batch_size=config.BATCH_SIZE, load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.
    Saves the result to submission.csv.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission on device: {device}")

    # Load Test Data
    test_loader = get_dataloader(
        "test", batch_size=batch_size, shuffle=False, load_cached_data=load_cached_data
    )

    # Load Model
    model = HCPDS().to(device)
    if not os.path.exists(config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {config.MODEL_CHECKPOINT}. Please run training first."
        )

    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_feat = batch["atomic_features"].to(device)
            global_feat = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"].cpu().numpy()

            outputs = model(atomic_feat, global_feat, mask)

            # Inverse transform: exp(y) - 1 (since targets were log1p)
            preds = torch.expm1(outputs).cpu().numpy()

            ids.extend(batch_ids)
            predictions.extend(preds)

    # Create DataFrame
    predictions = np.array(predictions)
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID as per sample submission
    submission_df.sort_values("id", inplace=True)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)

    # Save
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
