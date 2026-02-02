import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.utils import seed_everything, calculate_mcrmse


def train(config: Config = None):
    """
    Executes the training pipeline:
    1. Initializes model, optimizer, scheduler, loss.
    2. Runs training and validation loops.
    3. Saves the best model based on Validation MCRMSE.
    4. Implements Early Stopping.
    """
    if config is None:
        config = Config()

    # 1. Setup
    seed_everything(config.seed)
    device = torch.device(config.device)

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(config=config)

    # Model
    model = RNAModel(config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Loss Function
    criterion = MaskedMSELoss()

    # Tracking
    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.working_dir, "best_model.pth")
    patience = 5
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.num_epochs):
        # ================= TRAIN PHASE =================
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            # Move data to device
            seqs = batch["seq"].to(device)
            loops = batch["loop"].to(device)
            dists = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(seqs, loops, dists)

            # Compute Loss
            loss = criterion(preds, targets, masks)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(1, train_steps)

        # Update Scheduler
        scheduler.step()

        # ================= VALIDATION PHASE =================
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                seqs = batch["seq"].to(device)
                loops = batch["loop"].to(device)
                dists = batch["dist"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seqs, loops, dists)

                # Slice predictions and targets to the scored length (68) for metric calculation
                # Shape: (Batch, 68, 3)
                preds_scored = preds[:, : config.pred_len, :]
                targets_scored = targets[:, : config.pred_len, :]

                val_preds_list.append(preds_scored.cpu())
                val_targets_list.append(targets_scored.cpu())

        # Concatenate all validation batches
        val_preds_all = torch.cat(val_preds_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)

        # Compute MCRMSE
        val_mcrmse = calculate_mcrmse(val_preds_all, val_targets_all).item()

        # ================= LOGGING & CHECKPOINTING =================
        print(
            f"Epoch {epoch + 1:02d}/{config.num_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f}",
            end="",
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(" [Saved Best]")
            patience_counter = 0
        else:
            print("")
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    print(f"Training finished. Best Val MCRMSE: {best_mcrmse:.6f}")
    return best_model_path


def generate_submission(config: Config = None):
    """
    Generates the submission file using the best trained model.
    1. Loads best model.
    2. Predicts on Test set.
    3. Maps predictions to sample_submission format.
    4. Saves CSV.
    """
    if config is None:
        config = Config()

    device = torch.device(config.device)
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Best model not found. Cannot generate submission.")
        return

    # Load Model
    model = RNAModel(config).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Test Data
    _, _, test_loader = get_dataloaders(config=config)

    # Store predictions: {sample_id: tensor_of_shape(107, 3)}
    preds_map = {}

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            seqs = batch["seq"].to(device)
            loops = batch["loop"].to(device)
            dists = batch["dist"].to(device)
            ids = batch["id"]

            # Forward pass (Batch, 107, 3)
            preds = model(seqs, loops, dists)
            preds = preds.cpu().numpy()

            for i, sample_id in enumerate(ids):
                preds_map[sample_id] = preds[i]

    # Load Sample Submission
    print(f"Reading sample submission from {config.sample_submission_file}...")
    sub_df = pd.read_csv(config.sample_submission_file)

    # Columns to update
    # config.target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # We need to map these to the columns in the CSV.
    # The CSV columns are: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Create arrays for the new values
    reactivity_vals = []
    deg_Mg_pH10_vals = []
    deg_pH10_vals = []  # Not predicted, fill 0
    deg_Mg_50C_vals = []
    deg_50C_vals = []  # Not predicted, fill 0

    # Iterate through submission rows
    # This assumes the submission file is well-formed.
    # We parse 'id_seqpos' to look up our predictions.

    # Optimization: Process by parsing ID and seqpos
    # id_seqpos format: "id_00073f8be_0"

    for idx, row in sub_df.iterrows():
        id_seqpos = row["id_seqpos"]
        # Split from the right to handle underscores in ID if any (though IDs usually don't have them)
        # Format is {id}_{seqpos}
        sample_id, seqpos_str = id_seqpos.rsplit("_", 1)
        seqpos = int(seqpos_str)

        if sample_id in preds_map:
            # Get prediction vector for this position: [reactivity, deg_Mg_pH10, deg_Mg_50C]
            # Note: We must ensure the order matches config.target_cols
            # config.target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
            pred_vec = preds_map[sample_id][seqpos]

            reactivity_vals.append(pred_vec[0])
            deg_Mg_pH10_vals.append(pred_vec[1])
            deg_Mg_50C_vals.append(pred_vec[2])
        else:
            # Fallback if ID not found (should not happen)
            reactivity_vals.append(0.0)
            deg_Mg_pH10_vals.append(0.0)
            deg_Mg_50C_vals.append(0.0)

        # Fill unscored columns
        deg_pH10_vals.append(0.0)
        deg_50C_vals.append(0.0)

    # Update DataFrame
    sub_df["reactivity"] = reactivity_vals
    sub_df["deg_Mg_pH10"] = deg_Mg_pH10_vals
    sub_df["deg_pH10"] = deg_pH10_vals
    sub_df["deg_Mg_50C"] = deg_Mg_50C_vals
    sub_df["deg_50C"] = deg_50C_vals

    # Save
    print(f"Saving submission to {config.submission_file}...")
    sub_df.to_csv(config.submission_file, index=False)
    print("Submission saved successfully.")


def run_training_pipeline():
    """
    Helper function to run the full pipeline.
    """
    config = Config()
    train(config)
    generate_submission(config)
