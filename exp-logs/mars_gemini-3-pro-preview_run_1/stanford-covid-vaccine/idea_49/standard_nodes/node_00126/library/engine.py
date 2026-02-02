import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from library.config import Config
from library.dataset import RNADataset
from library.model import SpectralTopologicalBiGRU
from library.utils import masked_mse_loss, mcrmse_loss


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fn(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move batch to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        lpe = batch["lpe"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequence, loop_type, pair_dist, lpe)

        # Compute loss (Masked MSE)
        loss = masked_mse_loss(targets, outputs, mask)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimization step
        optimizer.step()

        # Accumulate loss (weighted by batch size for correct average)
        running_loss += loss.item() * sequence.size(0)
        count += sequence.size(0)

    return running_loss / count


def eval_fn(model, loader, device):
    """
    Evaluates the model on the validation set and computes MCRMSE.
    Aggregates all predictions first to compute the global metric.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            lpe = batch["lpe"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(sequence, loop_type, pair_dist, lpe)

            all_preds.append(outputs)
            all_targets.append(targets)
            all_masks.append(mask)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Compute MCRMSE
    score = mcrmse_loss(all_targets, all_preds, all_masks)

    return score.item()


def generate_submission(device):
    """
    Loads the best model, predicts on the test set, and generates the submission CSV.
    """
    print("Generating submission...")

    # Load Test Data
    test_dataset = RNADataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = SpectralTopologicalBiGRU()
    model.to(device)

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Inference
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            lpe = batch["lpe"].to(device)

            # Forward
            outputs = model(sequence, loop_type, pair_dist, lpe)

            # Store predictions (move to CPU numpy)
            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(batch["id"])

    # Concatenate predictions: Shape (N_samples, 107, 3)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare Submission Data
    submission_data = []

    # Target columns in the model output: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Target columns in submission: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            # Row ID format: id_{id}_{seqpos}
            row_id = f"{sample_id}_{seqpos}"

            # Extract predicted values
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    # Create DataFrame
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_data, columns=cols)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main execution function for training and validation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Initializing training on {device}...")

    # 2. Data Loading
    train_dataset = RNADataset(mode="train")
    val_dataset = RNADataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SpectralTopologicalBiGRU()
    model.to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, device)

        # Validate
        val_score = eval_fn(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging (Full Precision)
        print(
            f"Epoch {epoch + 1} | LR: {current_lr:.2e} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> Model Saved. New Best MCRMSE: {best_score}")

    print(f"Training finished. Best Validation MCRMSE: {best_score}")

    # 6. Generate Submission
    generate_submission(device)
