import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.train import train_model
from library.dataset import get_train_val_datasets, get_test_dataset
from library.inference import optimize_threshold, generate_predictions


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train Model
    # We override max_epochs to 10 to ensure the run completes well within the 2-hour limit
    # while providing enough convergence for the baseline.
    print("Starting Model Training...")
    model, _ = train_model(debug=False, max_epochs=10, batch_size=Config.BATCH_SIZE)

    # 3. Validation and Threshold Optimization
    print("\nPerforming Validation and Threshold Optimization...")
    # Load validation dataset (uses cached parquet if available)
    _, val_dataset = get_train_val_datasets(debug=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Optimize threshold
    best_threshold, final_mcc = optimize_threshold(model, val_loader, device)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {final_mcc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    model.eval()

    # Collect predictions and targets
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            x_kin, x_vis = inputs
            x_kin = x_kin.to(device, non_blocking=True)
            x_vis = x_vis.to(device, non_blocking=True)

            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits).view(-1)

            all_probs.append(probs.cpu())
            all_targets.append(targets.view(-1).cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()

    # Calculate absolute error
    errors = np.abs(y_true - y_prob)

    # Load feature data for correlation analysis
    # We read the cached validation parquet file to get feature values and names
    if os.path.exists(Config.CACHE_VAL_FEATURES):
        val_df = pd.read_parquet(Config.CACHE_VAL_FEATURES)

        # Identify feature columns (exclude metadata)
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        feature_cols = [c for c in val_df.columns if c not in meta_cols]

        # Ensure alignment (parquet vs dataset order is preserved)
        if len(val_df) == len(errors):
            # Compute correlations
            # We create a temporary series for errors to correlate against the dataframe
            # Using a loop or apply might be slow, so we use pandas corrwith or just manual iteration for top features
            # To be memory efficient and fast, we'll just iterate over columns

            correlations = {}
            error_series = pd.Series(errors)

            for col in feature_cols:
                # Simple correlation
                corr = val_df[col].corr(error_series)
                correlations[col] = corr

            corr_series = pd.Series(correlations)

            print(
                "Top 5 features positively correlated with error (Error increases as feature increases):"
            )
            print(corr_series.sort_values(ascending=False).head(5))

            print(
                "\nTop 5 features negatively correlated with error (Error decreases as feature increases):"
            )
            print(corr_series.sort_values(ascending=True).head(5))
        else:
            print(
                "Mismatch between validation dataframe and prediction length. Skipping detailed failure analysis."
            )
    else:
        print("Validation cache file not found. Skipping failure analysis.")

    # 5. Submission
    TARGET_SCORE = 0.6634847318478787

    if final_mcc > TARGET_SCORE:
        print(
            f"\nValidation Score ({final_mcc}) > Target ({TARGET_SCORE}). Generating submission..."
        )

        # Load Test Data
        test_dataset, test_ids = get_test_dataset()

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate and Save Predictions
        generate_predictions(model, test_loader, test_ids, best_threshold, device)

    else:
        print(
            f"\nValidation Score ({final_mcc}) <= Target ({TARGET_SCORE}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
