import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import load_data
from library.model import DualStreamBiGRU


def train_fn(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move inputs to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["targets"].to(device)

        batch_size = seq.size(0)

        optimizer.zero_grad()

        inputs = {"seq": seq, "loop": loop, "dist": dist}
        outputs = model(inputs)  # (B, L, 3)

        # Masked MSE: Only first 68 positions contribute to loss
        outputs_masked = outputs[:, : Config.PRED_LEN, :]
        targets_masked = targets[:, : Config.PRED_LEN, :]

        loss = criterion(outputs_masked, targets_masked)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)

            inputs = {"seq": seq, "loop": loop, "dist": dist}
            outputs = model(inputs)  # (B, L, 3)

            # Slice to scored positions for metric calculation
            outputs_masked = outputs[:, : Config.PRED_LEN, :]
            targets_masked = targets[:, : Config.PRED_LEN, :]

            preds_list.append(outputs_masked.cpu().numpy())
            targets_list.append(targets_masked.cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Compute MCRMSE
    score = mcrmse(targets, preds)
    return score


def predict_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            inputs = {"seq": seq, "loop": loop, "dist": dist}
            outputs = model(inputs)  # (B, L, 3)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    preds = np.concatenate(preds_list, axis=0)  # (N, 107, 3)
    return preds, ids_list


def run_training(debug=False):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)
    Config.create_dirs()

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    train_dataset = load_data("train", debug=debug)
    val_dataset = load_data("val", debug=debug)

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

    # Initialize Model
    model = DualStreamBiGRU()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    best_score = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_score = eval_fn(model, val_loader, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! Score: {best_score:.6f}")

    print(f"Training Complete. Best Validation Score: {best_score:.6f}")
    return best_model_path


def generate_submission(model_path, debug=False):
    """
    Generates submission file using the best model.
    """
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_dataset = load_data("test", debug=debug)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = DualStreamBiGRU()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Generating predictions on test set...")
    preds, ids = predict_fn(model, test_loader, device)
    # preds shape: (N_samples, 107, 3)

    submission_data = []

    # Map predictions to columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Output columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Predicted values
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Unscored/Untrained values (fill with 0)
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
    sub_df = pd.DataFrame(submission_data, columns=columns)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    print(f"Submission shape: {sub_df.shape}")
