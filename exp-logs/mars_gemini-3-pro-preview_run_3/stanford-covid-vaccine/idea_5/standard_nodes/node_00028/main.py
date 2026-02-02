import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, calculate_metric, create_submission_file
from library.data_loader import get_dataloaders
from library.trainer import Trainer


def main():
    # 1. Setup
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Configure for fast baseline execution
    # Reducing epochs to ensure completion within time limits while allowing convergence
    Config.EPOCHS = 25

    # Update submission path to meet prompt requirements
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    print("Initializing Data Loaders...")
    # Load data (using cache if available)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model Training
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 3. Validation and Failure Analysis
    print("\n" + "=" * 40)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Load best model weights
    if os.path.exists(Config.MODEL_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        )
    trainer.model.eval()

    # Collect Validation Predictions
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for inputs, targets, ids in val_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = trainer.model(inputs)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric
    final_metric = calculate_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Metadata Features
    # Calculate RMSE per sample on scored columns
    scored_indices = Config.SCORED_COLS_INDICES
    seq_scored = Config.PRED_LEN  # 68

    # Slice to scored region and columns
    p_scored = val_preds[:, :seq_scored, scored_indices]
    t_scored = val_targets[:, :seq_scored, scored_indices]

    # Mean Squared Error per sample (averaging over position and channel)
    mse_per_sample = np.mean((p_scored - t_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Validation Metadata to get features
    val_meta_df = pd.read_parquet(Config.VAL_METADATA)

    # Align metadata with predictions using IDs
    # Create a mapping from ID to error
    error_df = pd.DataFrame({"id": val_ids, "error": rmse_per_sample})
    analysis_df = pd.merge(val_meta_df, error_df, on="id", how="inner")

    # Calculate GC Content
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )

    # Features to analyze
    features = ["signal_to_noise", "SN_filter", "gc_content"]

    print("\nCorrelation between Model Error and Input Features:")
    print(f"{'Feature':<20} {'Correlation':<12} {'P-Value':<12}")
    print("-" * 44)

    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs if any (though data analysis showed none)
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data[feat], valid_data["error"])
                print(f"{feat:<20} {corr:<12.4f} {p_val:<12.4e}")
            else:
                print(f"{feat:<20} {'N/A':<12} {'N/A':<12}")

    # 4. Submission
    THRESHOLD = 0.7421537041664124

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold (< {THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        # Note: trainer.predict loads the best model internally, but we already loaded it.
        # However, to be safe and use the method as designed:
        test_preds = trainer.predict(test_loader)

        # Get Test IDs for submission file creation
        test_ids = []
        for _, ids in test_loader:
            test_ids.extend(ids)

        # Create submission file
        create_submission_file(test_preds, test_ids)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} does not meet threshold (< {THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
