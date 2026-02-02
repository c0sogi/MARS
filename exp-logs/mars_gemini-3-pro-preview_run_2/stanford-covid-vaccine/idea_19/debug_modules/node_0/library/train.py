import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config, set_seed
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMCRMSELoss
from library.utils import GlobalMCRMSE


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, partner_indices, targets in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, partner_indices)

        # Compute Masked Loss (only on scored columns)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model and computes the Global MCRMSE on scored columns.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    metric = GlobalMCRMSE()

    # Determine indices of scored columns for metric calculation
    # Config.ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, t in enumerate(Config.ALL_TARGETS) if t in Config.SCORED_TARGETS
    ]

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            batch_size = inputs.size(0)
            dataset_size += batch_size

            # Forward pass
            outputs = model(inputs, partner_indices)

            # Loss tracking (Masked Loss)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * batch_size

            # Metric Update
            # Filter outputs and targets to only keep scored columns for the metric
            outputs_scored = outputs[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            metric.update(outputs_scored, targets_scored)

    epoch_loss = running_loss / dataset_size
    epoch_mcrmse = metric.compute()

    return epoch_loss, epoch_mcrmse


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()

    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            outputs = model(inputs, partner_indices)

            # Move to CPU and numpy
            preds_np = outputs.cpu().numpy()  # (Batch, SeqLen, 5)
            all_preds.append(preds_np)

    # Concatenate all batches
    # Shape: (N_Samples, SeqLen, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Load Test Metadata to get IDs
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    ids = test_df["id"].values

    # Prepare data for DataFrame
    submission_data = []

    # Iterate through samples
    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (SeqLen, 5)

        # Iterate through sequence positions
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            # Get values for the 5 columns
            # Order in model output matches Config.ALL_TARGETS:
            # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            vals = sample_preds[seqpos]

            row_dict = {
                "id_seqpos": row_id,
                "reactivity": vals[0],
                "deg_Mg_pH10": vals[1],
                "deg_pH10": vals[2],
                "deg_Mg_50C": vals[3],
                "deg_50C": vals[4],
            }
            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False, epochs=None):
    """
    Main training driver.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override epochs if provided
    num_epochs = epochs if epochs is not None else Config.EPOCHS

    # Load Data
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Initialize Model
    print("Initializing Model...")
    model = RNAModel().to(device)

    # Loss and Optimizer
    criterion = MaskedMCRMSELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.FACTOR,
        patience=Config.PATIENCE,
        verbose=True,
    )

    # Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stop_counter = 0
    early_stop_patience = Config.PATIENCE + 2  # Slightly more lax than scheduler

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MCRMSE: {best_mcrmse:.10f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load Best Model for Inference
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
