import pandas as pd
import numpy as np
import gc
import os
import sys

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.feature_builder import FeatureBuilder
from library.model_factory import LGBMClassifierWrapper
from library.metrics import optimize_threshold


def run():
    # 1. Setup
    print("Setting up configuration...")
    Config.setup()

    # 2. Build Features
    # Initialize FeatureBuilder to create windowed tabular data
    builder = FeatureBuilder()

    print("Building/Loading Training Features...")
    df_train = builder.build_features(split="train", load_cached_data=True)

    print("Building/Loading Validation Features...")
    df_val = builder.build_features(split="val", load_cached_data=True)

    # 3. Prepare Data for Model
    # Define metadata columns to exclude from training features
    # These are identifiers or non-predictive metadata
    metadata_cols = [
        "contact_id",
        "game_play",
        "step",
        "datetime",
        "contact",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
        "nfl_player_id_2_int",  # Intermediate column potentially left by builder
    ]

    # Identify feature columns dynamically
    # Exclude known metadata and any string-based paths that might have slipped through
    drop_cols = [c for c in metadata_cols if c in df_train.columns]
    feature_cols = [
        c
        for c in df_train.columns
        if c not in drop_cols and not c.startswith("video_path")
    ]

    print(f"Selected {len(feature_cols)} features for training.")

    # Full Training: Use all available data to maximize information density
    # (Cite solution_lesson_node_00003)
    print(f"Using full training set: {len(df_train)} samples.")

    X_train = df_train[feature_cols]
    y_train = df_train["contact"]

    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    # Clean up df_train to free memory
    del df_train
    gc.collect()

    # 4. Train Model
    print("Initializing and Training Model...")
    model_wrapper = LGBMClassifierWrapper()

    # Train using the wrapper which handles early stopping
    model_wrapper.train(X_train, y_train, X_val, y_val)

    # 5. Validation & Threshold Optimization
    print("Predicting on Validation Set...")
    val_probs = model_wrapper.predict(X_val)

    print("Optimizing Threshold...")
    best_threshold, best_mcc = optimize_threshold(y_val, val_probs)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {best_mcc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error magnitude (absolute difference between truth and probability)
    errors = np.abs(y_val - val_probs)

    # Calculate correlation between features and error magnitude
    # This helps identify which features are associated with hard-to-predict samples
    feature_corrs = X_val.corrwith(pd.Series(errors, index=X_val.index))

    print("Top 10 Features correlated with Error Magnitude:")
    print(feature_corrs.abs().sort_values(ascending=False).head(10))

    # 7. Inference on Test Set
    print("Building/Loading Test Features...")
    df_test = builder.build_features(split="test", load_cached_data=True)

    X_test = df_test[feature_cols]

    print("Predicting on Test Set...")
    test_probs = model_wrapper.predict(X_test)

    # Apply the optimized threshold found during validation
    test_preds = (test_probs >= best_threshold).astype(int)

    # 8. Submission
    BASELINE_MCC = 0.5733787587361774

    if best_mcc > BASELINE_MCC:
        print(
            f"Validation MCC ({best_mcc:.6f}) improved over baseline ({BASELINE_MCC:.6f}). Generating submission..."
        )
        print("Generating Submission File...")
        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": test_preds}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")
    else:
        print(
            f"Validation MCC ({best_mcc:.6f}) did not improve over baseline ({BASELINE_MCC:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    run()
