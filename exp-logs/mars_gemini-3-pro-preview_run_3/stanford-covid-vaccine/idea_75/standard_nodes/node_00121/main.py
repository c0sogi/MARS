import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.data import get_dataloaders
from library.train import Trainer


def analyze_failures(val_df, val_preds, val_targets, seq_scored=68):
    """
    Performs failure analysis by correlating model error with input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Calculate RMSE per sample
    # val_preds: (N, 107, 5), val_targets: (N, 107, 5)
    # Slice to scored region
    preds_sliced = val_preds[:, :seq_scored, :]
    targs_sliced = val_targets[:, :seq_scored, :]

    # Select scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    preds_scored = preds_sliced[:, :, scored_indices]
    targs_scored = targs_sliced[:, :, scored_indices]

    # MSE per sample (average over length and channels)
    mse_per_sample = np.mean((preds_scored - targs_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    val_df["rmse"] = rmse_per_sample

    # 2. Feature Engineering on Validation Metadata
    # Nucleotide content
    for char in ["A", "G", "C", "U"]:
        val_df[f"pct_{char}"] = val_df["sequence"].apply(
            lambda s: s.count(char) / len(s)
        )

    # Structure content
    val_df["pct_unpaired"] = val_df["structure"].apply(lambda s: s.count(".") / len(s))

    # 3. Correlation Analysis
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_unpaired",
    ]

    print(f"{'Feature':<20} {'Correlation with Error':<25}")
    print("-" * 45)

    for feat in features_to_check:
        if feat in val_df.columns:
            # Drop NaNs if any (though data should be clean)
            valid_data = val_df[[feat, "rmse"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                print(f"{feat:<20} {corr:.4f}")


def main():
    # 1. Configuration
    config = Config(debug=False)

    # Adjust config for fast baseline and submission requirements
    config.epochs = 20  # Limit epochs for speed
    config.submission_path = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    # Set seeds
    seed_everything(config.seed)

    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")

    # 2. Data Loading
    print("\nLoading Data...")
    # Using load_cached_data=True as requested
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Training
    print("\nInitializing Trainer...")
    trainer = Trainer(config)

    print("\nStarting Training...")
    trainer.fit(train_loader, val_loader)

    # 4. Validation & Analysis
    print("\nRunning Validation Inference...")
    # Load best model explicitly
    trainer.model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )
    trainer.model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for features, pair_indices, targets in val_loader:
            features = features.to(config.device)
            pair_indices = pair_indices.to(config.device)

            outputs = trainer.model(features, pair_indices)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Metric
    final_metric = metric_mcrmse(val_targets, val_preds, seq_scored=config.pred_len)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Load validation metadata to correlate errors with features
    val_df = pd.read_parquet(config.val_file)
    analyze_failures(val_df, val_preds, val_targets, seq_scored=config.pred_len)

    # 5. Submission
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test
        test_preds = trainer.predict(test_loader)

        # Generate Submission File
        trainer.generate_submission(test_preds)

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
