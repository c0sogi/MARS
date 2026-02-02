import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_dataloaders
from library.model import NRDCN
from library.loss import MCRMSELoss


def train_model():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = NRDCN().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = MCRMSELoss().to(device)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        start_time = time.time()

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # --- Stabilized Recurrent Loop ---

            # Pass 1: Cold Start (recycling=None -> zeros)
            pred_1 = model(inputs, partner_indices, recycling=None)

            # Detach for feedback to prevent noisy gradients looping back infinitely
            recycling_input = pred_1.detach()

            # Pass 2: Refinement
            pred_2 = model(inputs, partner_indices, recycling=recycling_input)

            # --- Loss Calculation ---
            # Primary loss on refined output, Auxiliary loss on cold start
            loss_2 = criterion(pred_2, targets)
            loss_1 = criterion(pred_1, targets)

            total_loss = loss_2 + 0.5 * loss_1

            total_loss.backward()
            optimizer.step()

            train_loss_accum += total_loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # 6. Validation
        model.eval()
        val_loss_accum = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                targets = batch["targets"].to(device)

                # Validation also uses the 2-pass strategy for best accuracy
                pred_1 = model(inputs, partner_indices, recycling=None)
                pred_2 = model(inputs, partner_indices, recycling=pred_1)

                # We calculate loss/metric on the refined prediction
                loss = criterion(pred_2, targets)
                val_loss_accum += loss.item()

                # Store for metric calculation (needs CPU numpy)
                # Slice predictions to match target length (68)
                seq_len_target = targets.shape[1]
                val_preds.append(pred_2[:, :seq_len_target, :].cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)

        # Flatten for metric calculation
        val_preds_cat = np.concatenate(val_preds, axis=0)
        val_targets_cat = np.concatenate(val_targets, axis=0)

        # Calculate MCRMSE on scored columns
        # Identify scored indices
        scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        val_mcrmse = mcrmse_metric(val_targets_cat, val_preds_cat, scored_indices)

        # Scheduler Step
        scheduler.step(val_mcrmse)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_val_loss:
            best_val_loss = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! MCRMSE: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 7. Inference on Test Set
    print("\nStarting Inference on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["id"]

            # 2-Pass Inference Strategy
            pred_1 = model(inputs, partner_indices, recycling=None)
            pred_2 = model(inputs, partner_indices, recycling=pred_1)

            # pred_2 shape: (Batch, 107, 5)
            preds_np = pred_2.cpu().numpy()

            batch_size, seq_len, num_targets = preds_np.shape

            for i in range(batch_size):
                sample_id = ids[i]
                sample_preds = preds_np[i]  # (107, 5)

                for seqpos in range(seq_len):
                    # Construct row ID: id_{id}_{seqpos}
                    row_id = f"{sample_id}_{seqpos}"

                    # Get predictions for this position
                    # Targets order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    vals = sample_preds[seqpos]

                    row_data = {
                        "id_seqpos": row_id,
                        "reactivity": vals[0],
                        "deg_Mg_pH10": vals[1],
                        "deg_pH10": vals[2],
                        "deg_Mg_50C": vals[3],
                        "deg_50C": vals[4],
                    }
                    submission_data.append(row_data)

    # 8. Save Submission
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = submission_df[cols]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
