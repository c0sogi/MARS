import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from library.config import Config, set_seed
from library.utils import get_device
from library.dataset import DataManager
from library.model import DisentangledTCNLSTM


class MaskedMAELoss(nn.Module):
    """
    Custom Loss function that computes Mean Absolute Error (L1)
    only for the inspiratory phase of the breath (where u_out == 0).
    """

    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, pred, target, u_out):
        """
        Args:
            pred (torch.Tensor): Predicted pressure, shape (Batch, 80)
            target (torch.Tensor): Actual pressure, shape (Batch, 80)
            u_out (torch.Tensor): Expiratory valve control, shape (Batch, 80)
        """
        # Create mask: 1 for inspiratory (u_out=0), 0 for expiratory (u_out=1)
        mask = 1 - u_out

        # Calculate element-wise L1 loss masked by the phase
        loss = torch.abs(pred - target) * mask

        # Normalize by the number of valid (inspiratory) time steps
        # Add a small epsilon to avoid division by zero in case of empty mask
        valid_points = mask.sum()
        if valid_points > 0:
            return loss.sum() / valid_points
        else:
            return loss.sum()


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    loss_fn = MaskedMAELoss()

    # Identify the index of 'u_out' in the skip connection features
    try:
        u_out_idx = Config.SKIP_FEATURES.index("u_out")
    except ValueError:
        raise ValueError("u_out feature missing from Config.SKIP_FEATURES")

    for inputs, targets in dataloader:
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)

        # Extract u_out for masking. inputs['skip'] is (Batch, Length, Features)
        u_out = inputs["skip"][:, :, u_out_idx]

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        # Compute loss
        loss = loss_fn(preds, targets, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_fn(model, dataloader, device):
    """
    Performs evaluation on the validation set.
    """
    model.eval()
    total_loss = 0.0
    loss_fn = MaskedMAELoss()

    try:
        u_out_idx = Config.SKIP_FEATURES.index("u_out")
    except ValueError:
        raise ValueError("u_out feature missing from Config.SKIP_FEATURES")

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            u_out = inputs["skip"][:, :, u_out_idx]

            preds = model(inputs)
            loss = loss_fn(preds, targets, u_out)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def run_experiment():
    """
    Main driver function to run the full training and submission pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # Ensure working and submission directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Running experiment on device: {device}")

    # 2. Data Preparation
    dm = DataManager()
    # Load dataloaders. DataManager handles caching internally.
    train_loader = dm.get_dataloader("train", shuffle=True, load_cached_data=True)
    val_loader = dm.get_dataloader("validation", shuffle=False, load_cached_data=True)
    test_loader = dm.get_dataloader("test", shuffle=False, load_cached_data=True)

    # 3. Model Initialization
    model = DisentangledTCNLSTM().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_loss = eval_fn(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Print metrics (full precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Inference and Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches: (N_batches, B, 80) -> (N_total_breaths, 80)
    predictions = np.concatenate(all_preds, axis=0)
    # Flatten to match the submission format (row-wise)
    flat_preds = predictions.flatten()

    # Load test metadata to ensure ID alignment
    print("Generating submission file...")
    test_df = pd.read_csv(Config.TEST_PATH)

    # The DataManager sorts data by breath_id and time_step.
    # We must sort the test dataframe identically to align predictions with IDs.
    test_df.sort_values(["breath_id", "time_step"], inplace=True)

    submission = pd.DataFrame({"id": test_df["id"], "pressure": flat_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
