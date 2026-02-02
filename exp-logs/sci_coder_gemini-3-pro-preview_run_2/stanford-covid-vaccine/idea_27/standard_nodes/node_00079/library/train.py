import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import RNANet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, partner_indices, partner_mask, targets) in enumerate(
        loader
    ):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        partner_mask = partner_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, partner_indices, partner_mask)

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the MCRMSE score.
    """
    model.eval()
    running_loss = 0.0
    metric = MCRMSEMetric()

    with torch.no_grad():
        for inputs, partner_indices, partner_mask, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            partner_mask = partner_mask.to(device)
            targets = targets.to(device)

            preds = model(inputs, partner_indices, partner_mask)

            loss = criterion(preds, targets)
            running_loss += loss.item()

            # Update metric tracker
            metric.update(preds, targets)

    avg_loss = running_loss / len(loader)
    mcrmse_score = metric.compute()

    return avg_loss, mcrmse_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a numpy array of shape (N_Samples, Seq_Len, Num_Targets).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices, partner_mask, _ in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            partner_mask = partner_mask.to(device)

            preds = model(inputs, partner_indices, partner_mask)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def generate_submission(model, test_loader, device):
    """
    Generates predictions and saves the submission file.
    """
    print("Generating predictions for test set...")

    # Get raw predictions: (240, 107, 5)
    preds = predict(model, test_loader, device)

    # Load test.csv to get IDs
    test_df = pd.read_csv(Config.TEST_CSV)
    ids = test_df["id"].values

    # Prepare data for submission dataframe
    submission_data = []

    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


def train_model():
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 2. Model Setup
    model = RNANet().to(device)
    criterion = MaskedMCRMSELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 3. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )  # Full precision

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  [+] Saved best model (MCRMSE: {best_mcrmse})")
        else:
            patience_counter += 1
            print(
                f"  [-] No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")

    # 4. Inference
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    generate_submission(model, test_loader, device)


def run_training():
    train_model()
