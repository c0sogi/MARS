import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc

# Import from provided library files
from library.config import Config
from library.data_utils import get_data
from library.dataset import get_dataloaders
from library.model import AS_DFRN
from library.loss import MCRMSELoss

# =============================================================================
# Configuration & Setup
# =============================================================================
SEED = 42
MAX_EPOCHS = 15  # Limited for fast baseline execution
THRESHOLD_METRIC = 0.47142532743789534


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # Data Loading
    # =========================================================================
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # =========================================================================
    # Model Initialization
    # =========================================================================
    print("Initializing AS-DFRN model...")
    model = AS_DFRN().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Loss Function (Training uses Boundary Anchoring: full 107 length)
    criterion = MCRMSELoss().to(device)

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"Starting training for {MAX_EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, pair_indices, targets, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward Pass (Iterative Refinement)
            # y_pred_1: Pass 1 (Zero Feedback)
            # y_pred_2: Pass 2 (Dense Feedback)
            y_pred_1, y_pred_2 = model(inputs, pair_indices)

            # Loss Calculation
            # L_total = L(Y2) + 0.5 * L(Y1)
            loss_2 = criterion(y_pred_2, targets)
            loss_1 = criterion(y_pred_1, targets)
            loss = loss_2 + 0.5 * loss_1

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation Step (Fast approximation for scheduler)
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for inputs, pair_indices, targets, _ in val_loader:
                inputs = inputs.to(device)
                pair_indices = pair_indices.to(device)
                targets = targets.to(device)

                _, y_pred_2 = model(inputs, pair_indices)
                # Use same criterion for consistency in scheduler monitoring
                val_loss = criterion(y_pred_2, targets)
                val_loss_accum += val_loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss (Approx): {avg_val_loss:.4f} | Time: {elapsed:.1f}s"
        )

        # Save best model state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # =========================================================================
    # Final Validation & Metric Calculation
    # =========================================================================
    print("\nPerforming Final Validation...")
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model.eval()

    # Metric Accumulators
    # We need to compute MCRMSE on the first 68 positions for scored columns [0, 1, 3]
    scored_cols = Config.SCORED_INDICES
    sse_per_col = np.zeros(len(scored_cols))
    count_per_col = 0

    # Store per-sample errors for failure analysis
    sample_ids = []
    sample_errors = []

    with torch.no_grad():
        for inputs, pair_indices, targets, ids in val_loader:
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)
            targets = targets.to(device)

            # Inference
            _, y_pred = model(inputs, pair_indices)

            # Move to CPU
            y_pred = y_pred.cpu().numpy()
            targets = targets.cpu().numpy()

            # Slice to Scorable Length (68) and Scored Columns
            # y_pred shape: (B, 107, 5)
            y_pred_scored = y_pred[:, : Config.SCORABLE_LENGTH, scored_cols]
            targets_scored = targets[:, : Config.SCORABLE_LENGTH, scored_cols]

            # Accumulate SSE globally
            # (B, 68, 3) -> squared diff
            sq_diff = (y_pred_scored - targets_scored) ** 2

            # Sum over batch and sequence length
            sse_per_col += np.sum(sq_diff, axis=(0, 1))
            count_per_col += y_pred_scored.shape[0] * y_pred_scored.shape[1]

            # Per-sample error calculation (MCRMSE per sample)
            # Mean over sequence (axis 1) and columns (axis 2) -> sqrt
            # Note: This is an approximation for ranking; precise per-sample MCRMSE:
            mse_sample = np.mean(sq_diff, axis=1)  # (B, 3)
            rmse_sample = np.sqrt(mse_sample)  # (B, 3)
            mcrmse_sample = np.mean(rmse_sample, axis=1)  # (B,)

            sample_ids.extend(ids)
            sample_errors.extend(mcrmse_sample)

    # Compute Global MCRMSE
    # RMSE per column = sqrt(Sum_SSE / Total_Count)
    rmse_per_col = np.sqrt(sse_per_col / count_per_col)
    final_metric = np.mean(rmse_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # Failure Analysis
    # =========================================================================
    print("\nPerforming Failure Analysis...")
    # Load Validation Metadata
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if os.path.exists(val_csv_path):
        val_df = pd.read_csv(val_csv_path)

        # Create Error DataFrame
        error_df = pd.DataFrame({"id": sample_ids, "error": sample_errors})

        # Merge with metadata
        analysis_df = pd.merge(error_df, val_df, on="id", how="left")

        # Calculate Correlations
        if "signal_to_noise" in analysis_df.columns:
            corr_sn = analysis_df["error"].corr(analysis_df["signal_to_noise"])
            print(f"Correlation (Error vs Signal_to_Noise): {corr_sn:.4f}")

        if "SN_filter" in analysis_df.columns:
            corr_filter = analysis_df["error"].corr(analysis_df["SN_filter"])
            print(f"Correlation (Error vs SN_filter): {corr_filter:.4f}")

        if "mean_reactivity" in analysis_df.columns:
            corr_react = analysis_df["error"].corr(analysis_df["mean_reactivity"])
            print(f"Correlation (Error vs Mean Reactivity): {corr_react:.4f}")
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # =========================================================================
    # Submission Generation
    # =========================================================================
    if final_metric < THRESHOLD_METRIC:
        print(
            f"\nMetric {final_metric} < Threshold {THRESHOLD_METRIC}. Generating Submission..."
        )

        preds_list = []
        ids_list = []

        with torch.no_grad():
            for inputs, pair_indices, _, ids in test_loader:
                inputs = inputs.to(device)
                pair_indices = pair_indices.to(device)

                # Inference
                _, y_pred = model(inputs, pair_indices)

                # Move to CPU
                y_pred = y_pred.cpu().numpy()  # (B, 107, 5)

                preds_list.append(y_pred)
                ids_list.extend(ids)

        # Concatenate all predictions
        all_preds = np.concatenate(preds_list, axis=0)  # (N_test, 107, 5)

        # Flatten for submission format
        # We need to create rows for id_seqpos
        submission_data = []
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(ids_list):
            sample_pred = all_preds[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()
                submission_data.append([row_id] + row_values)

        # Create DataFrame
        sub_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + target_cols)

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} >= Threshold {THRESHOLD_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
