import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.data_processing import get_dataloaders
from library.models import SPIRVNet, FocalLoss
from library.train import (
    train_one_epoch,
    validate,
    find_optimal_threshold,
    predict_test,
)


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    # Limit epochs to ensure completion within time limit
    Config.EPOCHS = 6

    print(f"Running with Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # load_cached_data=True will use existing parquet files or create them
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    # Infer input dimensions from a batch
    dummy_kin, dummy_vis, _ = next(iter(train_loader))
    input_dim_kin = dummy_kin.shape[1]
    input_dim_vis = dummy_vis.shape[1]

    print(f"Input Dims - Kinematic: {input_dim_kin}, Visual: {input_dim_vis}")

    model = SPIRVNet(input_dim_kin, input_dim_vis).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_mcc = -1.0
    best_threshold = 0.5

    print("\n--- Starting Training ---")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Optimize Threshold
        curr_thresh, curr_mcc = find_optimal_threshold(val_targets, val_probs)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MCC: {curr_mcc:.5f}"
        )

        # Checkpoint
        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold = curr_thresh
            save_checkpoint(model, optimizer, epoch, best_mcc, Config.MODEL_SAVE_PATH)

    # =========================================================================
    # 5. Final Evaluation
    # =========================================================================
    print("\n--- Final Evaluation ---")
    # Load Best Model
    checkpoint = load_checkpoint(model, Config.MODEL_SAVE_PATH, device=Config.DEVICE)
    print(
        f"Loaded best model from Epoch {checkpoint['epoch']} (MCC: {checkpoint['score']:.5f})"
    )

    # Run Validation on Full Set with Best Model
    val_loss, val_probs, val_targets = validate(model, val_loader, criterion, device)

    # Re-optimize threshold to be precise
    final_threshold, final_mcc = find_optimal_threshold(val_targets, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")
    try:
        # Load feature names and raw values from the cached parquet file
        # This corresponds to the validation set
        if os.path.exists(Config.CACHE_VAL_FEATURES):
            df_val_features = pd.read_parquet(Config.CACHE_VAL_FEATURES)

            # Remove target column if present to isolate features
            if "contact" in df_val_features.columns:
                features_df = df_val_features.drop(columns=["contact"])
            else:
                features_df = df_val_features

            # Compute Error Magnitude
            # val_probs are from the best model
            # val_targets are ground truth
            errors = np.abs(val_targets - val_probs)

            # Ensure lengths match
            if len(errors) == len(features_df):
                correlations = []
                # Calculate correlation for numerical columns
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    # Fill NaNs with 0 for correlation calculation
                    feat_vals = features_df[col].fillna(0).values

                    # Compute Pearson correlation
                    if np.std(feat_vals) > 1e-9:  # Avoid constant columns
                        corr = np.corrcoef(feat_vals, errors)[0, 1]
                        if not np.isnan(corr):
                            correlations.append((col, corr))

                # Sort by absolute correlation
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)

                print("Top 10 Features Correlated with Error Magnitude:")
                for name, corr in correlations[:10]:
                    print(f"  {name}: {corr:.4f}")
            else:
                print(
                    f"Mismatch in lengths: Errors ({len(errors)}) vs Features ({len(features_df)})"
                )
        else:
            print(
                "Validation cache file not found. Skipping detailed failure analysis."
            )

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    TARGET_METRIC = 0.6634847318478787

    if final_mcc > TARGET_METRIC:
        print(f"\nPerformance Condition Met: {final_mcc} > {TARGET_METRIC}")
        print("Generating submission file...")

        # Generate predictions
        predictions = predict_test(model, test_loader, final_threshold, device)

        # Load metadata to map predictions to contact_ids
        df_test_meta = pd.read_csv(Config.METADATA_TEST)

        if len(predictions) != len(df_test_meta):
            print(
                f"Error: Prediction length {len(predictions)} != Metadata length {len(df_test_meta)}"
            )
        else:
            df_test_meta["contact"] = predictions
            submission_df = df_test_meta[["contact_id", "contact"]]

            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nPerformance Condition Not Met: {final_mcc} <= {TARGET_METRIC}")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
