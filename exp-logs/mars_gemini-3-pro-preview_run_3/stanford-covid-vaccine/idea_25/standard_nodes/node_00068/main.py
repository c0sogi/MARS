import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_global_mcrmse
from library.data import get_dataloaders
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failures(trainer, val_loader, val_parquet_path):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample RMSE and correlates it with metadata features.
    """
    print("\n==== Failure Analysis ====")

    # Load validation metadata
    val_df = pd.read_parquet(val_parquet_path)

    # Get predictions on validation set using the best model
    # Trainer.predict loads the best model state
    ids, preds = trainer.predict(val_loader)

    # Ensure alignment
    # The loader might shuffle or not, but Trainer.predict returns ids corresponding to preds.
    # We create a dataframe for preds to merge with val_df
    pred_df = pd.DataFrame({"id": ids})
    pred_df["preds"] = list(preds)  # Store arrays in cells

    # Merge with ground truth df
    merged_df = pd.merge(val_df, pred_df, on="id", how="inner")

    # Extract Targets and Preds for scoring
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # We need to compute RMSE per sample over the scored sequence length (68)
    seq_scored = trainer.config.seq_scored

    # Use scored columns for analysis as well
    scored_indices = getattr(trainer.config, "scored_cols_indices", [0, 1, 3])

    sample_errors = []

    for _, row in merged_df.iterrows():
        # Stack targets: (68, 5)
        # Note: targets in dataframe are lists of length 68
        t_list = [row[c] for c in target_cols]
        # Transpose to (68, 5)
        targets = np.array(t_list).T

        # Get preds: (107, 5) -> slice to (68, 5)
        p = row["preds"][:seq_scored]

        # Filter for scored columns for fair analysis
        targets = targets[:, scored_indices]
        p = p[:, scored_indices]

        # Compute RMSE for this sample
        # Mean over positions and targets, then sqrt
        mse = np.mean((targets - p) ** 2)
        rmse = np.sqrt(mse)
        sample_errors.append(rmse)

    merged_df["error_rmse"] = sample_errors

    # Feature Engineering for Correlation
    # 1. Signal to Noise
    # 2. SN_filter
    # 3. GC Content

    def calculate_gc(seq):
        return (seq.count("G") + seq.count("C")) / len(seq)

    merged_df["gc_content"] = merged_df["sequence"].apply(calculate_gc)

    # Calculate Correlations
    corr_features = ["signal_to_noise", "SN_filter", "gc_content"]
    correlations = merged_df[corr_features].corrwith(merged_df["error_rmse"])

    print("Correlation between Error (RMSE) and Features:")
    print(correlations)

    # Identify worst performing samples
    print("\nTop 5 Worst Performing Samples (Highest RMSE):")
    print(merged_df.nlargest(5, "error_rmse")[["id", "error_rmse", "signal_to_noise"]])


def main():
    # 1. Configuration
    config = Config()

    # Explicitly patch scored_cols_indices to ensure it exists even if module reload fails
    config.scored_cols_indices = [0, 1, 3]

    # Adjust config for fast baseline execution if necessary
    # The provided config has 20 epochs. On A100 with ~2k samples, this is very fast (<5 mins).
    # We will stick to the default config to ensure convergence, but ensure output path is correct.

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    config.submission_path = os.path.join(submission_dir, "submission.csv")

    # Set seeds
    seed_everything(config.seed)

    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")

    # 2. Data Loading
    print("\nLoading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Training
    print("\nInitializing Trainer...")
    trainer = Trainer(config)

    print("\nStarting Training...")
    trainer.fit(train_loader, val_loader)

    # 4. Validation & Metric
    print("\nCalculating Final Validation Metric...")
    # Load best model implicitly via validate if we wanted, but validate uses current model.
    # However, fit() saves the best model. We should load it to be precise.
    # Trainer.validate uses self.model.
    # Let's manually load the best state dict into the trainer's model for final validation
    if os.path.exists(config.model_save_path):
        trainer.model.load_state_dict(
            torch.load(config.model_save_path, map_location=config.device)
        )
        print("Loaded best model checkpoint.")

    val_score = trainer.validate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    analyze_failures(trainer, val_loader, config.val_data_path)

    # 6. Submission
    threshold = 0.5978901386
    if val_score < threshold:
        print(
            f"\nValidation score ({val_score}) meets threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        ids, preds = trainer.predict(test_loader)

        # Generate Submission File
        trainer.generate_submission(ids, preds)

        # Verify file creation
        if os.path.exists(config.submission_path):
            print(f"Submission successfully saved to {config.submission_path}")
        else:
            print("Error: Submission file not found after generation.")
    else:
        print(
            f"\nValidation score ({val_score}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
