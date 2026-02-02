import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import ConvBiGRU


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Accumulate weighted loss
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)  # Shape: (Batch, 107, 5)

            # Slice to scored length (first 68 positions)
            outputs = outputs[:, : Config.PRED_LEN, :]  # Shape: (Batch, 68, 5)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate all predictions
    preds_arr = np.concatenate(preds_list, axis=0)  # Shape: (N_samples, 68, 5)

    # Prepare data for DataFrame
    n_samples = preds_arr.shape[0]
    seq_len = preds_arr.shape[1]  # Should be 68

    # Create id_seqpos column
    # Repeat each ID 68 times
    ids_repeated = np.repeat(ids_list, seq_len)
    # Tile positions 0..67 N times
    pos_tiled = np.tile(np.arange(seq_len), n_samples)

    id_seqpos = [f"{i}_{p}" for i, p in zip(ids_repeated, pos_tiled)]

    # Flatten predictions to (N_samples * 68, 5)
    preds_flat = preds_arr.reshape(-1, Config.OUTPUT_DIM)

    # Define columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Create DataFrame
    df_sub = pd.DataFrame(preds_flat, columns=target_cols)
    df_sub.insert(0, "id_seqpos", id_seqpos)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    model = ConvBiGRU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Loss Functions
    # Train on all columns to learn shared physics
    criterion_train = MCRMSELoss(select_columns=None)

    # Validate only on scored columns to match competition metric
    # Scored indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    criterion_val = MCRMSELoss(select_columns=Config.SCORED_TARGET_INDICES)

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_train, device
        )
        val_loss = validate(model, val_loader, criterion_val, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MCRMSE: {train_loss} | Val Scored MCRMSE: {val_loss}"
        )

        # Early Stopping and Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 7. Submission
    # Load best model
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run_training()
