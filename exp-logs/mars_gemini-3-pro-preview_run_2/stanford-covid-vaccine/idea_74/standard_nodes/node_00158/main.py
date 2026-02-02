import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from library.config import Config, set_seed
from library.data import get_loaders, process_data
from library.model import HCHSGFN
from library.train import Trainer, generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample MCRMSE and correlates it with metadata features.
    """
    print("\n=== Failure Analysis ===")

    # Load validation metadata to get features like signal_to_noise
    val_metadata_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_metadata_path):
        print("Validation metadata not found. Skipping failure analysis.")
        return

    val_df = pd.read_csv(val_metadata_path)

    # Ensure model is in eval mode
    model.eval()

    sample_errors = []

    with torch.no_grad():
        for inputs, pairs, targets in val_loader:
            inputs = inputs.to(device)
            pairs = pairs.to(device)
            targets = targets.to(device)

            # Inference: Two passes (Static -> Feedback)
            pred1, _ = model(inputs, pairs, y_prev=None)
            pred2, _ = model(inputs, pairs, y_prev=pred1)

            # Slice to scored region and columns
            # Shape: (Batch, 68, 3)
            pred_scored = pred2[:, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES]
            target_scored = targets[
                :, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES
            ]

            # Calculate RMSE per sample
            # MSE per sample: mean over length(68) and columns(3)
            mse_per_sample = torch.mean((pred_scored - target_scored) ** 2, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            sample_errors.extend(rmse_per_sample.cpu().numpy())

    sample_errors = np.array(sample_errors)

    # Add errors to dataframe (assuming order is preserved, which it is for val_loader)
    if len(sample_errors) != len(val_df):
        print(
            f"Warning: Mismatch in validation samples. Metadata: {len(val_df)}, Loader: {len(sample_errors)}"
        )
        return

    val_df["model_error"] = sample_errors

    # Feature Engineering for Analysis
    # 1. Signal to Noise
    if "signal_to_noise" in val_df.columns:
        sn_corr, _ = stats.pearsonr(val_df["signal_to_noise"], val_df["model_error"])
        print(f"Correlation (Error vs Signal-to-Noise): {sn_corr:.4f}")

    # 2. GC Content
    if "sequence" in val_df.columns:
        val_df["gc_content"] = val_df["sequence"].apply(
            lambda s: (s.count("G") + s.count("C")) / len(s)
        )
        gc_corr, _ = stats.pearsonr(val_df["gc_content"], val_df["model_error"])
        print(f"Correlation (Error vs GC Content):      {gc_corr:.4f}")

    # 3. Mean Reactivity (if available, usually proxy for signal strength)
    if "mean_reactivity" in val_df.columns:
        mr_corr, _ = stats.pearsonr(val_df["mean_reactivity"], val_df["model_error"])
        print(f"Correlation (Error vs Mean Reactivity): {mr_corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to leverage preprocessed .npz files
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = HCHSGFN().to(device)

    # 4. Training
    # We use the Trainer class from library.train
    # Overriding epochs if necessary, but Config.EPOCHS (20) is appropriate for this dataset size.
    trainer = Trainer(model, device, train_loader, val_loader)

    print("Starting Training...")
    best_val_score = trainer.fit(epochs=Config.EPOCHS)

    # 5. Final Metrics
    # Printing full precision as requested
    print(f"Final Validation Metric: {best_val_score}")

    # 6. Failure Analysis
    # Load the best model state for analysis
    model.load_state_dict(torch.load(trainer.model_save_path, map_location=device))
    analyze_failures(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.47142532743789534

    if best_val_score < THRESHOLD:
        print(
            f"\nValidation score ({best_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Retrieve test IDs from cache
        data_cache = process_data(load_cached_data=True)
        test_ids = data_cache["test"]["ids"]

        submission_path = "./submission/submission.csv"
        generate_submission(
            trainer.model_save_path, test_loader, test_ids, submission_path
        )
    else:
        print(
            f"\nValidation score ({best_val_score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
