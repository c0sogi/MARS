import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import mcrmse_loss, set_seed
from library.data import get_loader
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        seq = batch["sequence"].to(device)
        loop = batch["loop_type"].to(device)
        pair = batch["pair_offset"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        preds = model(seq, loop, pair)

        # Masked MSE: Only calculate loss for the first 68 positions (Config.pred_len)
        preds_scored = preds[:, : Config.pred_len, :]
        targets_scored = targets[:, : Config.pred_len, :]

        loss = criterion(preds_scored, targets_scored)
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set and returns MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            pair = batch["pair_offset"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, pair)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    score = mcrmse_loss(all_targets, all_preds).item()
    return score


def train_model(epochs=Config.epochs, patience=5, load_cached_data=True):
    """
    Main training loop with Early Stopping.
    """
    set_seed()
    device = Config.device
    print(f"Device: {device}")

    # Data Loading
    train_loader = get_loader("train", shuffle=True, load_cached_data=load_cached_data)
    val_loader = get_loader("val", shuffle=False, load_cached_data=load_cached_data)

    # Model Initialization
    model = RNAModel().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.min_lr
    )
    criterion = nn.MSELoss()

    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs (Patience: {patience})...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Early Stopping & Model Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved to {Config.model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_score


def generate_submission(load_cached_data=True):
    """
    Generates submission file using the best saved model.
    """
    device = Config.device

    if not os.path.exists(Config.model_save_path):
        print(f"Error: Model file not found at {Config.model_save_path}")
        return

    print("Loading best model for inference...")
    model = RNAModel().to(device)
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.eval()

    test_loader = get_loader("test", shuffle=False, load_cached_data=load_cached_data)

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            pair = batch["pair_offset"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, pair)

            ids_list.extend(ids)
            preds_list.append(preds.cpu().numpy())

    preds_array = np.concatenate(preds_list, axis=0)  # Shape: (N_samples, 107, 3)

    print("Formatting submission...")
    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_array[i]

        for j in range(Config.seq_len):
            row_id = f"{sample_id}_{j}"

            # Extract predictions for the 3 scored targets
            reactivity = float(sample_preds[j, 0])
            deg_Mg_pH10 = float(sample_preds[j, 1])
            deg_Mg_50C = float(sample_preds[j, 2])

            # Unscored targets are set to 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    df_sub = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )

    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
