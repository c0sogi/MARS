import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import WideResBiGRU
from library.loss import MaskedMSELoss


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequence, loop_type, pair_dist)

        # Compute masked loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * sequence.size(0)

    # Step scheduler if it's per-epoch (CosineAnnealing usually stepped per epoch)
    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["target"].to(device)

            outputs = model(sequence, loop_type, pair_dist)

            # We only score on the first PRED_LEN positions for the metric
            # The mcrmse utility handles shape mismatch if we pass full sequence,
            # but strictly speaking, the metric is defined on the scored columns.
            # Let's slice to scored length for metric calculation to be precise.
            scored_len = Config.PRED_LEN

            # Move to CPU for numpy conversion
            preds_np = outputs[:, :scored_len, :].cpu().numpy()
            targets_np = targets[:, :scored_len, :].cpu().numpy()

            all_preds.append(preds_np)
            all_targets.append(targets_np)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = mcrmse(all_targets, all_preds)
    return score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(sequence, loop_type, pair_dist)

            # outputs shape: [Batch, SeqLen, 3]
            preds_np = outputs.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds_np)

    all_preds = np.concatenate(preds_list, axis=0)  # Shape: (N_samples, 107, 3)

    # Prepare submission data
    # We need to flatten: (N_samples * 107) rows
    submission_data = []

    seq_len = Config.SEQ_LEN

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"

            # Model predicts: reactivity, deg_Mg_pH10, deg_Mg_50C
            reactivity = sample_preds[pos, 0]
            deg_Mg_pH10 = sample_preds[pos, 1]
            deg_Mg_50C = sample_preds[pos, 2]

            # Unscored columns filled with 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    df_sub = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = df_sub[cols]

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Model
    model = WideResBiGRU().to(device)

    # 4. Optimizer & Scheduler & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    criterion = MaskedMSELoss(scoring_length=Config.PRED_LEN)

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_mcrmse}")

    print(f"Training complete. Best MCRMSE: {best_mcrmse}")

    # 6. Inference
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run_training()
