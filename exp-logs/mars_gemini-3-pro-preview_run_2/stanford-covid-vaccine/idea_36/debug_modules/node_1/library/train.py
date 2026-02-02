import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_loader
from library.model import CF_DCN


def train_one_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch with the Iterative Refinement Loop.
    """
    model.train()
    running_loss = 0.0

    # Scored indices for loss calculation
    scored_indices = Config.SCORED_TARGET_INDICES

    for inputs, partner_indices, targets in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 1. Compute Static Backbone Features (Z)
        # Shape: (Batch, Seq, Embed_Dim)
        z = model.forward_backbone(inputs)

        batch_size, seq_len, _ = z.shape

        # 2. Pass 1: Zero Feedback
        # Initialize feedback with zeros
        initial_preds = torch.zeros(
            batch_size, seq_len, Config.NUM_TARGETS, device=device, dtype=inputs.dtype
        )

        # Compute Y_1
        y_1 = model.forward_head(z, initial_preds, partner_indices)

        # 3. Pass 2: Feedback from Pass 1
        # Detach gradients from Y_1 to treat it as fixed input for the next step
        y_1_detached = y_1.detach()

        # Compute Y_2
        y_2 = model.forward_head(z, y_1_detached, partner_indices)

        # 4. Compute Loss
        # Loss is calculated only on scored columns
        loss_1 = mcrmse_loss(y_1, targets, scored_indices)
        loss_2 = mcrmse_loss(y_2, targets, scored_indices)

        # Composite Loss: Main prediction + Auxiliary supervision on first pass
        total_loss = loss_2 + Config.AUX_WEIGHT * loss_1

        # 5. Optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    scored_indices = Config.SCORED_TARGET_INDICES

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Use the full inference forward pass (encapsulates the 2 passes)
            preds = model(inputs, partner_indices)

            loss = mcrmse_loss(preds, targets, scored_indices)
            running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    # Load Test Data
    test_loader = get_loader(mode="test", load_cached_data=True)

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices, targets in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            preds = model(inputs, partner_indices)

            # Move to CPU
            preds = preds.cpu().numpy()

            # Collect IDs (targets in test loader are dummy, but IDs are valid)
            # The dataset returns (inputs, partner_indices, targets),
            # but we need access to IDs which are stored in the dataset object.
            # However, the DataLoader batches don't directly yield IDs.
            # We need to iterate the dataset indices or modify loader.
            # The provided `get_loader` returns a loader wrapping `RNADataset`.
            # `RNADataset` has `self.ids`.
            # Since shuffle=False, we can just access ids from the dataset directly
            # corresponding to the batch indices.
            pass

            all_preds.append(preds)

    # Concatenate all predictions: (Total_Samples, Seq_Len, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Retrieve IDs from the dataset
    dataset_ids = test_loader.dataset.ids

    # Prepare data for DataFrame
    submission_data = []

    # Column names for submission
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(dataset_ids):
        sample_preds = all_preds[i]  # Shape (107, 5)

        for seq_pos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos]

            # Create a dictionary for the row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training(debug=False, num_epochs=None):
    """
    Main driver function to train the model and generate submission.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    train_loader = get_loader(mode="train", debug=debug)
    val_loader = get_loader(mode="val", debug=debug)

    # 2. Model Setup
    model = CF_DCN().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=False
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print(f"  New best model saved! Loss: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")

    # 4. Load Best Model for Submission
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model weights.")

    # 5. Generate Submission
    generate_submission(model, device)
