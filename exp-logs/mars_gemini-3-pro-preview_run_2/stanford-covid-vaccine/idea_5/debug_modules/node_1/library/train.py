import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.loss import MCRMSELoss
from library.model import HybridNet, get_dataloaders


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    val_loss = running_loss / len(loader.dataset)
    return val_loss


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model, Criterion, Optimizer
    print("Initializing Model...")
    model = HybridNet().to(device)
    criterion = MCRMSELoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train MCRMSE: {train_loss} - Val MCRMSE: {val_loss}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {Config.PATIENCE} epochs of no improvement."
            )
            break

    print(f"Training complete. Best Validation MCRMSE: {best_val_loss}")


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the best trained model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data (Test Loader)
    # We re-call get_dataloaders to ensure consistency, though we only need test_loader
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Load Model
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Run training first."
        )

    model = HybridNet().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 3. Inference
    all_preds = []
    all_ids = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            ids = batch["ids"]

            # Forward pass
            # Output shape: (Batch, Seq_Len, 5)
            outputs = model(inputs)

            # Move to CPU and numpy
            preds = outputs.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate all predictions: (Total_Samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # 4. Format Submission
    # We need to flatten the predictions to (Total_Samples * 107, 5)
    # and create the corresponding id_seqpos keys.

    submission_ids = []

    # The target columns in the order output by the model
    # Config.ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    target_cols = Config.ALL_TARGET_COLS

    # Prepare lists for DataFrame construction
    flat_preds = []

    num_samples, seq_len, num_targets = all_preds.shape

    for i in range(num_samples):
        sample_id = all_ids[i]
        for j in range(seq_len):
            submission_ids.append(f"{sample_id}_{j}")
            flat_preds.append(all_preds[i, j, :])

    flat_preds = np.array(flat_preds)

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=target_cols)
    submission_df.insert(0, "id_seqpos", submission_ids)

    # 5. Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
