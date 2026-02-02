import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.model import DKRHNet
from library.data_utils import get_transformed_data


class MaskedL1Loss(nn.Module):
    """
    Computes Mean Absolute Error strictly on the inspiratory phase.
    The inspiratory phase is defined where u_out == 0.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        # u_out is at index 1 in Config.FEATURE_COLS
        self.u_out_idx = 1

    def forward(self, pred, target, input_tensor):
        """
        Args:
            pred: (batch, seq_len)
            target: (batch, seq_len)
            input_tensor: (batch, seq_len, features)
        """
        # Extract u_out. u_out=1 is expiratory, u_out=0 is inspiratory.
        # We want to score only when u_out == 0.
        u_out = input_tensor[:, :, self.u_out_idx]

        # Create mask: 1 where u_out == 0, else 0
        mask = 1 - u_out

        # Calculate absolute error
        loss = torch.abs(pred - target)

        # Apply mask
        loss = loss * mask

        # Compute mean over valid elements
        # Add epsilon to avoid division by zero (though unlikely in this dataset)
        score = loss.sum() / (mask.sum() + 1e-8)

        return score


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        # Compute masked loss
        loss = criterion(preds, targets, inputs)

        # Backpropagation
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization Step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets, inputs)

            running_loss += loss.item()

    return running_loss / len(loader)


def predict(model, loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            predictions.append(preds.cpu().numpy())

    # Concatenate all batches: (N_batches, batch_size, seq_len) -> (N_total, seq_len)
    predictions = np.concatenate(predictions, axis=0)

    # Flatten to (N_total * seq_len,)
    return predictions.flatten()


def train_model():
    """
    Main driver function to train the DKRH-Net model.
    Handles data loading, model initialization, training loop, early stopping, and saving.
    """
    # Set seed for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_transformed_data(load_cached_data=True)

    # 2. Initialize Model
    model = DKRHNet().to(device)

    # 3. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = MaskedL1Loss()

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> New best model saved (Val Loss: {val_loss})")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # 5. Generate Submission
    generate_submission(test_loader, device, best_model_path)


def generate_submission(test_loader, device, model_path):
    """
    Loads the best model, generates predictions on the test set, and saves the submission file.
    """
    print("Generating submission...")

    # Load Model
    model = DKRHNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Predict
    flat_preds = predict(model, test_loader, device)

    # Load IDs
    # IDs are cached in the working directory as per data_utils logic
    ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    if os.path.exists(ids_path):
        flat_ids = np.load(ids_path)
    else:
        # Fallback if cache missing (should not happen if train_model ran)
        # We would need to reload test csv, but assuming flow is correct:
        raise FileNotFoundError(
            f"Test IDs not found at {ids_path}. Run data processing first."
        )

    # Ensure shapes match
    if len(flat_ids) != len(flat_preds):
        print(
            f"Warning: Shape mismatch. IDs: {len(flat_ids)}, Preds: {len(flat_preds)}"
        )
        # Truncate to minimum length to allow saving, though this indicates an error
        min_len = min(len(flat_ids), len(flat_preds))
        flat_ids = flat_ids[:min_len]
        flat_preds = flat_preds[:min_len]

    # Create DataFrame
    submission_df = pd.DataFrame({"id": flat_ids, "pressure": flat_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
