import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, MCRMSELoss, GlobalMCRMSE
from library.data import get_dataloaders
from library.model import RHIDFN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch using the iterative refinement strategy.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass returns predictions from both passes
        # y1: Pass 1 (Zero Feedback)
        # y2: Pass 2 (Recycled Feedback from detached y1)
        y1, y2 = model(inputs, partner_indices)

        # Calculate combined loss
        # Note: The criterion handles masking of unscored columns and positions
        loss_1 = criterion(y1, targets)
        loss_2 = criterion(y2, targets)

        total_loss = (loss_2 * Config.LOSS_WEIGHT_PASS_2) + (
            loss_1 * Config.LOSS_WEIGHT_PASS_1
        )

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global MCRMSE.
    """
    model.eval()
    metric_calc = GlobalMCRMSE()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # We only evaluate the final refined prediction (y2)
            _, y2 = model(inputs, partner_indices)

            metric_calc.update(y2, targets)

    return metric_calc.compute()


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []
    all_ids = []

    # Inference Loop
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["ids"]

            # Use refined predictions
            _, y2 = model(inputs, partner_indices)

            # Move to CPU
            all_preds.append(y2.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches: (Num_Samples, Seq_Len, Num_Targets)
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        return

    # Prepare data for CSV
    # Flattening: We need one row per sequence position
    data_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            # Construct ID: id_seqpos
            row_id = f"{sample_id}_{seqpos}"

            # Get values for this position
            values = sample_preds[seqpos].tolist()

            row = [row_id] + values
            data_rows.append(row)

    # Create DataFrame
    columns = ["id_seqpos"] + target_cols
    df = pd.DataFrame(data_rows, columns=columns)

    # Save
    df.to_csv(output_path, index=False)


def run_training(debug=False, epochs=Config.NUM_EPOCHS):
    """
    Main entry point for the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model & Optimization
    model = RHIDFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Logging (Full precision for validation score)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with MCRMSE: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    generate_submission(model, test_loader, device, submission_path)
    print(f"Submission saved to {submission_path}")
