import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.utils import seed_everything, get_device, mcrmse
from library.data import get_dataloaders
from library.model import RNARegressor


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    # MSE Loss with reduction='none' to apply mask manually
    criterion = nn.MSELoss(reduction="none")

    for batch in loader:
        # Move batch to device
        seq_ids = batch["seq_ids"].to(device)
        loop_ids = batch["loop_ids"].to(device)
        pair_emb = batch["pair_emb"].to(device)
        pos_emb = batch["pos_emb"].to(device)
        targets = batch["targets"].to(device)  # (B, 107, 3)
        mask = batch["mask"].to(device)  # (B, 107)

        optimizer.zero_grad()

        # Forward pass
        preds = model(seq_ids, loop_ids, pair_emb, pos_emb)  # (B, 107, 3)

        # Calculate element-wise squared error
        loss_elementwise = criterion(preds, targets)  # (B, 107, 3)

        # Apply mask: Expand mask to cover channel dimension
        # mask is 1.0 for scored positions, 0.0 otherwise
        mask_expanded = mask.unsqueeze(-1)  # (B, 107, 1)

        masked_loss = loss_elementwise * mask_expanded

        # Compute mean loss over valid elements
        # Sum of valid elements: sum(mask) * num_channels
        valid_elements = mask_expanded.sum() * Config.NUM_CLASSES
        loss = masked_loss.sum() / (valid_elements + 1e-8)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * seq_ids.size(0)
        total_samples += seq_ids.size(0)

    return running_loss / total_samples


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq_ids = batch["seq_ids"].to(device)
            loop_ids = batch["loop_ids"].to(device)
            pair_emb = batch["pair_emb"].to(device)
            pos_emb = batch["pos_emb"].to(device)
            targets = batch["targets"].cpu().numpy()

            preds = model(seq_ids, loop_ids, pair_emb, pos_emb)
            preds = preds.cpu().numpy()

            # Slice to scored length (68) for metric calculation
            # The targets are already aligned to this in data processing,
            # but the model outputs full 107 length.
            all_preds.append(preds[:, : Config.PRED_LEN, :])
            all_targets.append(targets[:, : Config.PRED_LEN, :])

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    score = mcrmse(all_targets, all_preds)
    return score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in loader:
            seq_ids = batch["seq_ids"].to(device)
            loop_ids = batch["loop_ids"].to(device)
            pair_emb = batch["pair_emb"].to(device)
            pos_emb = batch["pos_emb"].to(device)

            preds = model(seq_ids, loop_ids, pair_emb, pos_emb)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches: (N_samples, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # Load test IDs to construct the submission file
    df_test = pd.read_parquet(Config.TEST_FILE)

    # Adjust for debug subsampling
    if len(all_preds) < len(df_test):
        df_test = df_test.iloc[: len(all_preds)]

    ids = df_test["id"].values

    # Prepare data for DataFrame
    id_seqpos_list = []
    # Flatten predictions to (N_samples * 107, 3)
    flat_preds = all_preds.reshape(-1, 3)

    # Generate id_seqpos strings
    # Vectorized approach or list comprehension
    for sample_id in ids:
        for i in range(Config.SEQ_LEN):
            id_seqpos_list.append(f"{sample_id}_{i}")

    # Create Submission DataFrame
    sub_df = pd.DataFrame()
    sub_df["id_seqpos"] = id_seqpos_list

    # Map model outputs to columns
    # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
    sub_df["reactivity"] = flat_preds[:, 0]
    sub_df["deg_Mg_pH10"] = flat_preds[:, 1]

    # Fill unscored/untrained columns with 0.0
    sub_df["deg_pH10"] = 0.0

    sub_df["deg_Mg_50C"] = flat_preds[:, 2]
    sub_df["deg_50C"] = 0.0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False):
    """
    Main training loop.
    """
    # Initialize Config instance (creates directories, sets debug params)
    config_instance = Config(debug=debug)
    epochs = config_instance.EPOCHS

    seed_everything(Config.SEED)
    device = get_device()

    print(f"Device: {device}")
    print(f"Debug Mode: {debug}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Initialize Model
    model = RNARegressor().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = float("inf")

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Training complete. Best Validation Score: {best_score:.6f}")

    # Load best model for inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
