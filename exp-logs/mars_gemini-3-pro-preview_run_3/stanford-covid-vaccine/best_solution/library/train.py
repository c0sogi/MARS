import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, mcrmse_loss, competition_metric
from library.data import get_loaders
from library.model import RNAModel


def train_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, bpp_indices)

        # Calculate loss (MCRMSE on all 5 targets)
        loss = mcrmse_loss(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally before calculating metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, bpp_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate to form global tensors
    # Shape: (Total_Val_Samples, Seq_Len, 5)
    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate Loss (All 5 targets)
    val_loss = mcrmse_loss(global_preds, global_targets).item()

    # Calculate Competition Metric (Scored 3 targets)
    score = competition_metric(global_preds, global_targets)

    return val_loss, score


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            ids = batch["id"]

            preds = model(inputs, bpp_indices)

            # Move to CPU numpy
            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, Seq_Len, 5)
    predictions = np.concatenate(all_preds, axis=0)

    # Prepare data for DataFrame construction
    # We need to flatten: N_samples * Seq_Len rows
    n_samples, seq_len, n_targets = predictions.shape

    # Repeat IDs for each sequence position
    # id_list becomes [id1, id1, ..., id2, id2, ...]
    id_col = np.repeat(all_ids, seq_len)

    # Create seqpos indices: [0, 1, ..., 106, 0, 1, ...]
    seqpos_col = np.tile(np.arange(seq_len), n_samples)

    # Create id_seqpos column
    id_seqpos_col = [f"{i}_{s}" for i, s in zip(id_col, seqpos_col)]

    # Flatten predictions: (N_samples * Seq_Len, 5)
    flat_preds = predictions.reshape(-1, n_targets)

    # Create DataFrame
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    submission_df = pd.DataFrame(flat_preds, columns=target_cols)
    submission_df.insert(0, "id_seqpos", id_seqpos_col)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(debug=False):
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    print(f"Running training on device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Data
    train_loader, val_loader, test_loader = get_loaders(debug=debug)

    # 3. Model
    model = RNAModel(config=Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    # Adjust epochs if debug
    epochs = Config.DEBUG_EPOCHS if debug else Config.EPOCHS

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Competition Score: {score}"
        )

        # Checkpointing & Early Stopping
        if score < best_score:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    # Can toggle debug here or via Config
    run_training(debug=Config.DEBUG)
