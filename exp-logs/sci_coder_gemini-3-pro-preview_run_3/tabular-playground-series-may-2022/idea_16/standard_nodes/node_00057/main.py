import sys
import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.train_eval import (
    set_seed,
    train_model,
    evaluate,
    predict,
    generate_submission,
)


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set by correlating feature values
    with the magnitude of prediction error.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    # 1. Get Predictions and Targets
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)
            targets = batch["target"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits).squeeze(1)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)

    # 2. Calculate Error Magnitude
    errors = np.abs(targets - probs)

    # 3. Load Feature Data
    # We load the processed dataframe to get the feature values aligned with the loader
    # (val_loader is shuffle=False, so indices match)
    try:
        val_df = pd.read_parquet(Config.VAL_CACHE_PATH)
    except FileNotFoundError:
        print(
            "Cached validation data not found. Skipping detailed feature correlation."
        )
        return

    # Drop non-feature columns for correlation analysis
    cols_to_drop = ["id", "target", "source_path"]
    feature_df = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])

    # Add error column
    feature_df["error_magnitude"] = errors

    # 4. Compute Correlations
    correlations = feature_df.corr(numeric_only=True)["error_magnitude"].drop(
        "error_magnitude"
    )

    # Sort by absolute correlation
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("\nTop 10 Features correlated with Error Magnitude:")
    print(sorted_corr.head(10))

    # Print directionality for the top 3
    print("\nDirectionality of Top 3 Error Drivers:")
    for feat in sorted_corr.head(3).index:
        corr_val = correlations[feat]
        direction = "Positive" if corr_val > 0 else "Negative"
        print(f"{feat}: {corr_val:.4f} ({direction})")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.create_directories()

    # 2. Data Loading
    # Using full dataset to maximize score. A100 is fast enough for 640k rows.
    print("Loading data...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, max_samples=None
    )

    # 3. Training
    print("Starting training...")
    model = train_model(train_loader, val_loader, metadata, epochs=Config.EPOCHS)

    # 4. Validation Evaluation
    print("Evaluating on Validation set...")
    val_auc = evaluate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    perform_failure_analysis(model, val_loader, Config.DEVICE)

    # 6. Submission Logic
    # Threshold defined in task description
    THRESHOLD = 0.9971550270448856

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions
        test_probs = predict(model, test_loader, Config.DEVICE)

        # Save submission
        generate_submission(
            test_probs, Config.SAMPLE_SUBMISSION_PATH, Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
