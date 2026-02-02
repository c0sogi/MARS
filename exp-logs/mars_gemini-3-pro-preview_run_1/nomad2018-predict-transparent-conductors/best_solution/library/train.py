import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import time
from library.config import Config
from library.model import AMSP_DS_Net
from library.dataset import get_dataloader


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        atomic_features = batch["atomic_features"].to(device)
        global_features = batch["global_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        outputs = model(atomic_features, global_features, batch_indices)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and column-wise RMSLE metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, batch_indices)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    # Calculate Column-wise RMSLE
    # Since targets are already log1p transformed, RMSE on these values IS the RMSLE
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse_per_col = np.mean((all_preds - all_targets) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)

    metrics = {
        "val_loss": total_loss,
        "rmsle_formation": rmsle_per_col[0],
        "rmsle_bandgap": rmsle_per_col[1],
        "rmsle_mean": np.mean(rmsle_per_col),
    }

    return metrics


def predict_and_submit(model, scaler, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load test data using the scaler fitted on training data
    test_loader = get_dataloader(
        "test", batch_size=Config.BATCH_SIZE, shuffle=False, scaler=scaler
    )

    model.eval()
    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            batch_ids = batch["ids"]

            outputs = model(atomic_features, global_features, batch_indices)

            # Inverse transform: exp(x) - 1
            # Clamp to avoid numerical instability if predictions are very small negative numbers
            preds_original_scale = torch.expm1(outputs)
            preds_original_scale = torch.clamp(preds_original_scale, min=0.0)

            ids.extend(batch_ids)
            predictions.append(preds_original_scale.cpu().numpy())

    predictions = np.vstack(predictions)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to ensure consistency
    submission_df = submission_df.sort_values("id")

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission_df.head())


def run_training():
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Loading datasets...")
    train_loader, scaler = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, scaler=scaler
    )

    # 2. Model Initialization
    model = AMSP_DS_Net().to(device)

    # 3. Optimization Setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pt")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"RMSLE Form: {val_metrics['rmsle_formation']:.6f} | "
            f"RMSLE Band: {val_metrics['rmsle_bandgap']:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print("  -> New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 5. Load best model and generate submission
    print("Training complete. Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path))
    predict_and_submit(model, scaler, device)


if __name__ == "__main__":
    run_training()
