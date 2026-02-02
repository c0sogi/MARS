import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import AnchoredMCRMSELoss
from library.data import load_data
from library.model import ADFRN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, partner_indices, targets) in enumerate(loader):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward Pass
        # Returns predictions from Pass 1 (no feedback) and Pass 2 (with feedback)
        preds_pass1, preds_pass2 = model(inputs, partner_indices)

        # Compute Anchored Loss
        loss = criterion(preds_pass1, preds_pass2, targets)

        # Backward Pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, metric, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    metric.reset()

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward Pass
            preds_pass1, preds_pass2 = model(inputs, partner_indices)

            # Loss calculation (Anchored Loss on full sequence)
            loss = criterion(preds_pass1, preds_pass2, targets)
            running_loss += loss.item()

            # Metric calculation
            # We use the refined predictions (Pass 2) for scoring
            metric.update(preds_pass2, targets)

    avg_loss = running_loss / len(loader)
    mcrmse_score = metric.compute()

    return avg_loss, mcrmse_score


def generate_submission(model, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    # Load Test Data
    test_loader = load_data(mode="test", load_cached_data=True)

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for inputs, partner_indices in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward Pass
            # We only care about the final refined prediction (Pass 2)
            _, preds_pass2 = model(inputs, partner_indices)

            # Move to CPU
            preds_np = preds_pass2.cpu().numpy()  # (B, 107, 5)

            # Collect data
            # We need to associate predictions with IDs from the dataset
            # The loader batch size aligns with the inputs
            # Accessing IDs is tricky with standard DataLoader unless we modify it to return IDs.
            # However, the RNADataset stores IDs. The DataLoader yields batches.
            # The cleanest way without modifying the loader yield signature in a complex way
            # is to iterate the dataset indices or trust the order if shuffle=False.
            # load_data for test sets shuffle=False.

            preds_list.append(preds_np)

    # Concatenate all predictions
    all_preds = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 5)

    # Get IDs from the dataset directly
    # The test loader dataset is accessible
    all_ids = test_loader.dataset.ids

    # Prepare submission data
    submission_rows = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_preds = sample_preds[seq_pos]

            # Create dictionary for the row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Loading datasets...")
    train_loader = load_data(mode="train", load_cached_data=True)
    val_loader = load_data(mode="val", load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = ADFRN().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = AnchoredMCRMSELoss()
    metric = MCRMSEMetric()

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stop_patience = 7
    no_improve_epochs = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, metric, device)

        # Scheduler Step
        scheduler.step(val_mcrmse)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model! MCRMSE: {best_mcrmse}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early Stopping
        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    # 6. Submission
    # Load best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    generate_submission(model, device, submission_path)


if __name__ == "__main__":
    run_training()
