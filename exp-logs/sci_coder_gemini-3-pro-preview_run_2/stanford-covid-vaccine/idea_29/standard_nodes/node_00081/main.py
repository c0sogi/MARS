import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.stats as stats

# Import library modules
from library import config
from library.utils import set_seed, get_device, MCRMSELoss
from library.data import get_dataloaders
from library.model import SR_DCN


def main():
    # =========================================================================
    # 1. SETUP & CONFIGURATION
    # =========================================================================
    # Override config for fast baseline execution
    config.TRAIN_PARAMS["num_epochs"] = 15
    config.TRAIN_PARAMS["batch_size"] = 32
    config.TRAIN_PARAMS["debug"] = False  # Use full data for valid metric

    # Ensure submission directory exists
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    set_seed(42)
    device = get_device()

    # =========================================================================
    # 2. DATA LOADING
    # =========================================================================
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=config.TRAIN_PARAMS["debug"], load_cached_data=True
    )

    # =========================================================================
    # 3. MODEL INITIALIZATION
    # =========================================================================
    model = SR_DCN().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAIN_PARAMS["learning_rate"],
        weight_decay=config.TRAIN_PARAMS["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = MCRMSELoss()

    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    best_val_loss = float("inf")
    model_save_path = config.PATHS["MODEL_SAVE"]

    for epoch in range(config.TRAIN_PARAMS["num_epochs"]):
        model.train()

        for batch_idx, (inputs, partner_indices, targets, masks, ids) in enumerate(
            train_loader
        ):
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            batch_size, seq_len, _ = inputs.shape

            # --- Pass 1: Cold Start ---
            recycling_zero = torch.zeros(
                batch_size, seq_len, config.MODEL_PARAMS["num_targets"]
            ).to(device)
            input_pass1 = torch.cat([inputs, recycling_zero], dim=2)
            preds_1 = model(input_pass1, partner_indices)

            # --- Pass 2: Refinement (Stabilized Recycling) ---
            recycling_pass2 = preds_1.detach()  # Gradient Detachment
            input_pass2 = torch.cat([inputs, recycling_pass2], dim=2)
            preds_2 = model(input_pass2, partner_indices)

            # --- Loss ---
            loss_2 = criterion(preds_2, targets, masks)
            loss_1 = criterion(preds_1, targets, masks)
            loss = loss_2 + 0.5 * loss_1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation Step ---
        # We compute a quick validation metric for scheduler/early stopping
        # We will do the rigorous metric calculation after training
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, partner_indices, targets, masks, ids in val_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                bs, sl, _ = inputs.shape

                # Pass 1
                recycling_zero = torch.zeros(
                    bs, sl, config.MODEL_PARAMS["num_targets"]
                ).to(device)
                input_pass1 = torch.cat([inputs, recycling_zero], dim=2)
                preds_1 = model(input_pass1, partner_indices)

                # Pass 2
                input_pass2 = torch.cat([inputs, preds_1], dim=2)
                preds_2 = model(input_pass2, partner_indices)

                loss = criterion(preds_2, targets, masks)
                val_loss_accum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_accum / val_batches
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

    # =========================================================================
    # 5. FINAL EVALUATION & METRIC
    # =========================================================================
    # Load best model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Compute Global MCRMSE on Validation Set
    scored_indices = criterion.scored_indices
    num_scored = len(scored_indices)
    total_sse = torch.zeros(num_scored).to(device)
    total_count = torch.zeros(num_scored).to(device)

    # Store per-sample errors for failure analysis
    sample_errors = []
    sample_ids = []

    with torch.no_grad():
        for inputs, partner_indices, targets, masks, ids in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            bs, sl, _ = inputs.shape

            # Inference (Pass 1 -> Pass 2)
            recycling_zero = torch.zeros(bs, sl, config.MODEL_PARAMS["num_targets"]).to(
                device
            )
            input_pass1 = torch.cat([inputs, recycling_zero], dim=2)
            preds_1 = model(input_pass1, partner_indices)

            input_pass2 = torch.cat([inputs, preds_1], dim=2)
            preds_2 = model(input_pass2, partner_indices)

            # Global Metric Accumulation
            preds_scored = preds_2[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]
            mask_bool = masks.bool()

            for i in range(num_scored):
                p_col = preds_scored[:, :, i]
                t_col = targets_scored[:, :, i]
                valid_p = p_col[mask_bool]
                valid_t = t_col[mask_bool]

                sse = torch.sum((valid_p - valid_t) ** 2)
                count = valid_p.numel()

                total_sse[i] += sse
                total_count[i] += count

            # Per-sample Error Calculation (for Failure Analysis)
            # We calculate RMSE across all scored positions for each sample
            for b in range(bs):
                s_mask = mask_bool[b]
                if s_mask.sum() == 0:
                    continue

                s_p = preds_scored[b][s_mask]
                s_t = targets_scored[b][s_mask]

                # RMSE for this sample (averaged over positions and scored columns)
                s_mse = torch.mean((s_p - s_t) ** 2)
                s_rmse = torch.sqrt(s_mse).item()

                sample_errors.append(s_rmse)
                sample_ids.append(ids[b])

    # Compute Final Metric
    rmse_per_col = torch.sqrt(total_sse / (total_count + 1e-8))
    final_metric = torch.mean(rmse_per_col).item()

    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 6. FAILURE ANALYSIS
    # =========================================================================
    # Load validation metadata to correlate errors
    val_df = pd.read_csv(config.PATHS["VAL_CSV"])

    # Create dataframe of errors
    error_df = pd.DataFrame({"id": sample_ids, "error_rmse": sample_errors})

    # Merge with metadata
    analysis_df = pd.merge(error_df, val_df, on="id", how="inner")

    # Calculate correlations
    # We check Signal to Noise and Sequence Length (though length is const 107, so maybe just SN)
    if "signal_to_noise" in analysis_df.columns:
        corr_sn, _ = stats.spearmanr(
            analysis_df["error_rmse"], analysis_df["signal_to_noise"]
        )
        print(f"Correlation (Error vs Signal-to-Noise): {corr_sn:.4f}")

    if "SN_filter" in analysis_df.columns:
        corr_snf, _ = stats.spearmanr(
            analysis_df["error_rmse"], analysis_df["SN_filter"]
        )
        print(f"Correlation (Error vs SN_filter): {corr_snf:.4f}")

    # =========================================================================
    # 7. SUBMISSION GENERATION
    # =========================================================================
    THRESHOLD = 0.5417620723771521

    if final_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")
        results = []
        target_cols = config.DATA_CONFIG["target_cols"]

        with torch.no_grad():
            for inputs, partner_indices, _, masks, ids in test_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)

                bs, sl, _ = inputs.shape

                # Pass 1
                recycling_zero = torch.zeros(
                    bs, sl, config.MODEL_PARAMS["num_targets"]
                ).to(device)
                input_pass1 = torch.cat([inputs, recycling_zero], dim=2)
                preds_1 = model(input_pass1, partner_indices)

                # Pass 2
                input_pass2 = torch.cat([inputs, preds_1], dim=2)
                preds_2 = model(input_pass2, partner_indices)

                preds_np = preds_2.cpu().numpy()

                for i in range(bs):
                    sample_id = ids[i]
                    sample_preds = preds_np[i]

                    for seqpos in range(sl):
                        row_id = f"{sample_id}_{seqpos}"
                        vals = sample_preds[seqpos]

                        row_dict = {"id_seqpos": row_id}
                        for col_idx, col_name in enumerate(target_cols):
                            row_dict[col_name] = float(vals[col_idx])
                        results.append(row_dict)

        submission_df = pd.DataFrame(results)
        cols = ["id_seqpos"] + target_cols
        submission_df = submission_df[cols]
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
