import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.engine import Engine, set_seed
from library.data import get_loaders

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # Limit epochs for a fast baseline execution
    Config.EPOCHS = 15

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Initializing Fast Baseline Run (Epochs={Config.EPOCHS})...")

    # Initialize Engine
    engine = Engine()

    # Load Data (using cached data if available)
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 2. Training Loop
    best_val_loss = float("inf")
    best_val_metric = float("inf")
    patience = Config.PATIENCE
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = engine.train_epoch(train_loader)

        # Validate
        val_loss, val_mcrmse = engine.validate(val_loader)

        # Scheduler Step
        engine.scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpointing based on Val Loss (consistent with Engine logic)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metric = val_mcrmse  # Track the metric associated with best loss
            torch.save(engine.model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"  New best model saved! Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 3. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    engine.model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    # Re-run validation to confirm exact metric and get predictions for analysis
    final_loss, final_metric = engine.validate(val_loader)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Get Validation Predictions
    # Note: engine.predict returns concatenated numpy arrays (N_samples, Seq_Len, Channels)
    # But engine.predict expects a loader that yields 'id'. val_loader does this.
    val_preds, val_ids = engine.predict(val_loader)

    # We need the ground truth targets aligned with these predictions.
    # We can extract them from the loader.
    val_targets_list = []
    for batch in val_loader:
        val_targets_list.append(batch["targets"].numpy())
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Calculate RMSE per sample (averaged over columns and sequence length)
    # Shape: (N, L, C)
    squared_diff = (val_preds - val_targets) ** 2
    # Mean over Length (1) and Channels (2)
    mse_per_sample = np.mean(squared_diff, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Metadata to correlate with features
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure alignment: The loader and metadata should be aligned if not shuffled.
    # val_loader is shuffle=False.
    # We can verify by ID mapping, but assuming standard loader behavior:
    # Create a mapping from ID to RMSE
    id_to_rmse = dict(zip(val_ids, rmse_per_sample))

    # Map RMSE to the dataframe
    val_meta_df["model_rmse"] = val_meta_df["id"].map(id_to_rmse)

    # Drop NaNs if any alignment issues (shouldn't be)
    val_meta_df = val_meta_df.dropna(subset=["model_rmse"])

    # Correlation with Signal to Noise
    if "signal_to_noise" in val_meta_df.columns:
        corr, _ = pearsonr(val_meta_df["signal_to_noise"], val_meta_df["model_rmse"])
        print(f"Correlation between Error (RMSE) and Signal-to-Noise: {corr:.4f}")

    # Correlation with Sequence Length (though all are 107, good to check if variable)
    if val_meta_df["seq_length"].nunique() > 1:
        corr_len, _ = pearsonr(val_meta_df["seq_length"], val_meta_df["model_rmse"])
        print(f"Correlation between Error (RMSE) and Sequence Length: {corr_len:.4f}")

    # 4. Conditional Submission
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict Test Set
        test_preds, test_ids = engine.predict(test_loader)

        # Format Submission
        submission_data = []
        # Config.ALL_TARGETS contains the 5 columns required

        for i, sample_id in enumerate(test_ids):
            pred_matrix = test_preds[i]  # (107, 5)

            for pos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{pos}"
                row_vals = pred_matrix[pos].tolist()
                submission_data.append([row_id] + row_vals)

        columns = ["id_seqpos"] + Config.ALL_TARGETS
        sub_df = pd.DataFrame(submission_data, columns=columns)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
