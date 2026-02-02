import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import gc

# Import provided libraries
from library.config import Config
from library.data_processing import DataLoader
from library.feature_engineering import FeatureEngine
from library.models import UnifiedEnsemble
from library.evaluation import ThresholdOptimizer

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # Set random seeds for reproducibility
    np.random.seed(Config.SEED)
    try:
        torch.manual_seed(Config.SEED)
    except:
        pass

    print("Initializing Fast Baseline Pipeline...")

    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # We limit the training data size to ensure execution within 2 hours
    # as per the "Fast Baseline" requirement.
    # 250,000 samples is selected as a safe balance between speed and model performance.
    TRAIN_SAMPLE_SIZE = 250000

    loader = DataLoader()
    engine = FeatureEngine()
    ensemble = UnifiedEnsemble()

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Step 1/6] Loading Data...")

    # Load Metadata
    # Train: Sampled for speed
    df_train_meta = loader.load_metadata(split="train", sample_size=TRAIN_SAMPLE_SIZE)
    # Val: Full for accurate metric calculation
    df_val_meta = loader.load_metadata(split="val")
    # Test: Full for submission
    df_test_meta = loader.load_metadata(split="test")

    # Load Tracking Data
    df_train_tracking = loader.load_tracking(split="train")
    df_test_tracking = loader.load_tracking(split="test")

    # Merge Tracking Data
    # We use load_cached_data=True to utilize any pre-computed merges in ./working
    print("Merging Train Data...")
    df_train_merged = loader.merge_tracking_data(
        df_train_meta, df_train_tracking, "train_sample", load_cached_data=True
    )

    print("Merging Val Data...")
    df_val_merged = loader.merge_tracking_data(
        df_val_meta, df_train_tracking, "val", load_cached_data=True
    )

    print("Merging Test Data...")
    df_test_merged = loader.merge_tracking_data(
        df_test_meta, df_test_tracking, "test", load_cached_data=True
    )

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 2/6] Generating Features...")
    # Note: df_tracking is passed to compute topological features (graph metrics)

    # Train Features
    df_train_feats = engine.generate_features(
        df_train_merged, df_train_tracking, "train_sample", load_cached_data=True
    )

    # Val Features
    df_val_feats = engine.generate_features(
        df_val_merged, df_train_tracking, "val", load_cached_data=True
    )

    # Test Features
    df_test_feats = engine.generate_features(
        df_test_merged, df_test_tracking, "test", load_cached_data=True
    )

    # Garbage Collection to free memory before training
    del df_train_meta, df_val_meta, df_test_meta
    del df_train_tracking, df_test_tracking
    del df_train_merged, df_val_merged, df_test_merged
    gc.collect()

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 3/6] Training Models...")

    # GPU Configuration
    # Automatically detect GPU and update model parameters for acceleration
    if torch.cuda.is_available():
        print("GPU detected. Configuring models for GPU acceleration.")
        # Update LightGBM params
        ensemble.lgbm.params["device"] = "gpu"
        # Update XGBoost params (assuming version supports 'cuda' or 'gpu_hist')
        ensemble.xgb.params["device"] = "cuda"
        ensemble.xgb.params["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")

    # Train the ensemble
    ensemble.train(df_train_feats, df_val_feats, target_col="contact")

    # ---------------------------------------------------------
    # 5. Validation & Threshold Optimization
    # ---------------------------------------------------------
    print("\n[Step 4/6] Evaluating on Validation Set...")

    # Predict probabilities on validation set
    val_probs = ensemble.predict_proba(df_val_feats)
    y_val = df_val_feats["contact"].values

    # Optimize Threshold to maximize MCC
    optimizer = ThresholdOptimizer()
    best_thresh, best_mcc = optimizer.optimize(y_val, val_probs)

    # REQUIRED OUTPUT: Print final validation metric
    print(f"Final Validation Metric: {best_mcc}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 5/6] Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Compute correlations between error magnitude and features
    feature_cols = ensemble.feature_cols
    correlations = {}

    # Fill NaNs in features for correlation calculation (e.g. lag features at start of play)
    X_val_filled = df_val_feats[feature_cols].fillna(0)

    for col in feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(X_val_filled[col]):
            try:
                corr = np.corrcoef(errors, X_val_filled[col])[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception:
                continue

    # Sort correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print(
        "\nTop 5 Features correlated with Error (Positive - Error increases with feature):"
    )
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    print(
        "\nTop 5 Features correlated with Error (Negative - Error decreases with feature):"
    )
    for name, val in sorted_corr[-5:]:
        print(f"  {name}: {val:.4f}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    print("\n[Step 6/6] Generating Submission...")

    THRESHOLD_SCORE = 0.658992501127342

    if best_mcc > THRESHOLD_SCORE:
        print(
            f"Validation MCC ({best_mcc}) meets threshold ({THRESHOLD_SCORE}). Creating submission."
        )

        # Predict on Test
        test_probs = ensemble.predict_proba(df_test_feats)
        test_preds = (test_probs >= best_thresh).astype(int)

        # Create DataFrame
        submission = pd.DataFrame(
            {"contact_id": df_test_feats["contact_id"], "contact": test_preds}
        )

        # Save
        sub_path = Config.SUBMISSION_PATH
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Validation MCC ({best_mcc}) is below threshold ({THRESHOLD_SCORE}). Submission skipped."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
