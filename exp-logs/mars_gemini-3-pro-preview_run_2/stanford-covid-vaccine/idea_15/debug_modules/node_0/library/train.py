import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import seed_everything, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import DenseStackingHybridNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, neighbor_indices, targets) in enumerate(loader):
        inputs = inputs.to(device)
        neighbor_indices = neighbor_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, neighbor_indices)

        # Compute loss (MaskedMCRMSELoss handles column selection internally)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, tracker, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    tracker.reset()

    with torch.no_grad():
        for inputs, neighbor_indices, targets in loader:
            inputs = inputs.to(device)
            neighbor_indices = neighbor_indices.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs, neighbor_indices)

            # Compute batch loss for logging
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Update tracker for global metric calculation
            tracker.update(targets, outputs)

    avg_loss = running_loss / len(loader)
    global_mcrmse = tracker.compute()

    return avg_loss, global_mcrmse


def generate_submission(model, loader, test_ids, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    preds_list = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for inputs, neighbor_indices, _ in loader:
            inputs = inputs.to(device)
            neighbor_indices = neighbor_indices.to(device)

            # Forward pass
            outputs = model(inputs, neighbor_indices)

            # Move to CPU and numpy
            preds_list.append(outputs.cpu().numpy())

    # Concatenate all batches: (N_samples, SeqLen, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    submission_data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False):
    """
    Main execution function for training and inference.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_loaders(debug=debug)

    # 3. Model Initialization
    print("Initializing Dense-Stacking Hybrid Network...")
    model = DenseStackingHybridNet().to(device)

    # 4. Optimization
    criterion = MaskedMCRMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 5. Metric Tracker
    tracker = MetricTracker()

    # 6. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience = 10
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, tracker, device)

        # Scheduler Step
        scheduler.step(val_mcrmse)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! (MCRMSE: {best_mcrmse})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")

    # 7. Submission Generation
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    generate_submission(model, test_loader, test_ids, device, submission_path)


if __name__ == "__main__":
    # Run training with debug=False for full training
    run_training(debug=False)
