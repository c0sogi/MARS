import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from library
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_dataloaders
from library.model import NRDCN
from library.loss import MCRMSELoss


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing Model...")
    model = NRDCN().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = MCRMSELoss().to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_val_loss = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # --- Stabilized Recurrent Loop ---
            # Pass 1: Cold Start
            pred_1 = model(inputs, partner_indices, recycling=None)

            # Detach for feedback
            recycling_input = pred_1.detach()

            # Pass 2: Refinement
            pred_2 = model(inputs, partner_indices, recycling=recycling_input)

            # Loss Calculation
            loss_2 = criterion(pred_2, targets)
            loss_1 = criterion(pred_1, targets)

            total_loss = loss_2 + 0.5 * loss_1

            total_loss.backward()
            optimizer.step()

            train_loss_accum += total_loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                targets = batch["targets"].to(device)

                # 2-Pass Validation
                pred_1 = model(inputs, partner_indices, recycling=None)
                pred_2 = model(inputs, partner_indices, recycling=pred_1)

                # Store predictions and targets for metric calculation
                # Slice to target length (68)
                seq_len_target = targets.shape[1]
                val_preds.append(pred_2[:, :seq_len_target, :].cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        # Concatenate
        val_preds_cat = np.concatenate(val_preds, axis=0)
        val_targets_cat = np.concatenate(val_targets, axis=0)

        # Calculate Metric
        scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        val_mcrmse = mcrmse_metric(val_targets_cat, val_preds_cat, scored_indices)

        # Scheduler
        scheduler.step(val_mcrmse)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_val_loss:
            best_val_loss = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # --------------------------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nLoading best model for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    val_ids = []
    val_preds_final = []
    val_targets_final = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            pred_1 = model(inputs, partner_indices, recycling=None)
            pred_2 = model(inputs, partner_indices, recycling=pred_1)

            seq_len_target = targets.shape[1]
            val_preds_final.append(pred_2[:, :seq_len_target, :].cpu().numpy())
            val_targets_final.append(targets.cpu().numpy())
            val_ids.extend(ids)

    val_preds_cat = np.concatenate(val_preds_final, axis=0)
    val_targets_cat = np.concatenate(val_targets_final, axis=0)

    # Final Metric
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]
    final_metric = mcrmse_metric(val_targets_cat, val_preds_cat, scored_indices)

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate RMSE per sample
    # val_preds_cat: (N, 68, 5)
    # Filter scored columns
    preds_scored = val_preds_cat[:, :, scored_indices]
    targets_scored = val_targets_cat[:, :, scored_indices]

    # MSE per sample (average over seq_len and targets)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"id": val_ids, "rmse": rmse_per_sample})

    # Load Metadata
    if os.path.exists(Config.VAL_CSV):
        meta_df = pd.read_csv(Config.VAL_CSV)
        # Merge
        analysis_df = analysis_df.merge(
            meta_df[["id", "signal_to_noise", "SN_filter"]], on="id", how="left"
        )

        # Correlations
        if "signal_to_noise" in analysis_df.columns:
            corr_sn = analysis_df["rmse"].corr(analysis_df["signal_to_noise"])
            print(f"Correlation (RMSE vs Signal_to_Noise): {corr_sn:.4f}")

        if "SN_filter" in analysis_df.columns:
            corr_filter = analysis_df["rmse"].corr(analysis_df["SN_filter"])
            print(f"Correlation (RMSE vs SN_filter): {corr_filter:.4f}")
    else:
        print("Validation metadata not found, skipping correlation analysis.")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5417620723771521

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        submission_data = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                ids = batch["id"]

                # 2-Pass Inference
                pred_1 = model(inputs, partner_indices, recycling=None)
                pred_2 = model(inputs, partner_indices, recycling=pred_1)

                preds_np = pred_2.cpu().numpy()  # (Batch, 107, 5)
                batch_size, seq_len, _ = preds_np.shape

                for i in range(batch_size):
                    sample_id = ids[i]
                    sample_preds = preds_np[i]

                    for seqpos in range(seq_len):
                        row_id = f"{sample_id}_{seqpos}"
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

        submission_df = pd.DataFrame(submission_data)
        cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        submission_df = submission_df[cols]

        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
        print(f"Submission shape: {submission_df.shape}")
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
