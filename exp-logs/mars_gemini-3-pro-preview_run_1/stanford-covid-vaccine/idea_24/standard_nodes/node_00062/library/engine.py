import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, mcrmse, format_submission
from library.dataset import RNADataset
from library.model import ScalarAggregatedWideResBiGRU


def masked_mse_loss(preds, targets):
    """
    Calculates MSE loss only on the scored positions (first 68).
    Args:
        preds: (B, L, 3)
        targets: (B, L, 3)
    """
    # Slice to the scored length
    preds_scored = preds[:, : Config.PRED_LEN, :]
    targets_scored = targets[:, : Config.PRED_LEN, :]

    # MSE Loss
    loss = nn.MSELoss()(preds_scored, targets_scored)
    return loss


def train_fn(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Iterate over batches (no progress bar to keep output clean as requested)
    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        loop_types = batch["loop_type"].to(device)
        pair_dists = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)

        batch_size = sequences.size(0)

        optimizer.zero_grad()

        outputs = model(sequences, loop_types, pair_dists)

        loss = masked_mse_loss(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(sequences, loop_types, pair_dists)

            # Slice to scored length for metric calculation
            outputs_scored = outputs[:, : Config.PRED_LEN, :]
            targets_scored = targets[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = mcrmse(all_targets, all_preds)
    return score


def inference_fn(model, dataloader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            ids = batch["id"]

            outputs = model(sequences, loop_types, pair_dists)

            # Slice to scored length
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def run_training():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Initializing Datasets...")
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

    # --- Model Initialization ---
    print("Initializing Model...")
    model = ScalarAggregatedWideResBiGRU()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # --- Training Loop ---
    best_score = float("inf")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_score = eval_fn(model, val_loader, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            print(
                f"Validation score improved. Saving model to {Config.BEST_MODEL_PATH}"
            )
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training Complete. Best Val MCRMSE: {best_score}")

    # --- Inference on Test Set ---
    print("Starting Inference on Test Set...")

    # Load Best Model
    model = ScalarAggregatedWideResBiGRU()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)

    test_dataset = RNADataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    preds, test_ids = inference_fn(model, test_loader, device)

    # --- Submission ---
    print("Generating Submission File...")
    format_submission(preds, test_ids, Config.SUBMISSION_FILE)
