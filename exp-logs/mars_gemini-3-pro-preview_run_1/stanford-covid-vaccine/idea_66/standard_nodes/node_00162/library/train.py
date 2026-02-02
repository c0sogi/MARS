import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.data import get_dataloaders
from library.model import RNAModel
from library.utils import seed_everything, mcrmse_metric


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    # Strictly use MSE Loss as per idea specification
    criterion = nn.MSELoss()

    for batch in loader:
        # Move inputs to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        struct = batch["struct"].to(device)
        target = batch["target"].to(device)  # Shape: (B, 107, 3)

        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, loop, struct)  # Shape: (B, 107, 3)

        # Calculate Loss: Masked MSE (only first 68 positions)
        # We slice both predictions and targets to the scored length
        preds_scored = preds[:, : Config.PRED_LEN, :]
        targets_scored = target[:, : Config.PRED_LEN, :]

        loss = criterion(preds_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()

        running_loss += loss.item() * seq.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            struct = batch["struct"].to(device)
            target = batch["target"].to(device)

            preds = model(seq, loop, struct)

            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE metric
    # This function handles the slicing to PRED_LEN internally
    score = mcrmse_metric(all_preds, all_targets, num_scored=Config.PRED_LEN)

    return score


def generate_submission(model, loader, device, output_path):
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            struct = batch["struct"].to(device)
            ids = batch["id"]

            # Forward pass
            preds = model(seq, loop, struct)  # (B, 107, 3)
            preds = preds.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds)

    all_preds = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 3)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            # Row ID
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = pd.DataFrame(submission_data, columns=columns)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.initialize()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = RNAModel().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train MSE: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! Score: {best_score:.6f}")

    print(f"\nTraining Complete. Best Validation MCRMSE: {best_score:.6f}")

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    train()
