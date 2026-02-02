import sys
import os
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import SUBMISSION_PATH, SEED
from library.utils import seed_everything, compute_mcc
from library.data_loader import load_dataset
from library.feature_generator import generate_features
from library.model_handler import DualStreamGBDT, generate_submission


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Data Loading
    # We load the full datasets to ensure temporal features (lags) are calculated correctly.
    # The ModelHandler will handle undersampling for training speed.
    print("Loading datasets...")
    try:
        # Load Train
        train_df = load_dataset(mode="train", load_cached_data=True)
        # Load Validation
        val_df = load_dataset(mode="validation", load_cached_data=True)
        # Load Test
        test_df = load_dataset(mode="test", load_cached_data=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Feature Generation
    # This converts raw data into Stream A (Interaction) and Stream B (Impact) feature sets
    print("Generating features...")
    try:
        train_data = generate_features(train_df, mode="train", load_cached_data=True)
        val_data = generate_features(val_df, mode="validation", load_cached_data=True)
        test_data = generate_features(test_df, mode="test", load_cached_data=True)
    except Exception as e:
        print(f"Error generating features: {e}")
        return

    # 4. Model Training
    print("Training model...")
    model = DualStreamGBDT()

    # The model handles undersampling internally (10:1 ratio) to ensure fast training
    model.train(train_data, val_data)

    # 5. Validation & Metrics
    print("Evaluating on validation set...")
    val_preds = model.predict(val_data)

    # Reconstruct ground truth for validation from the streams
    y_true_a = val_data["stream_a"]["y"]
    ids_a = val_data["stream_a"]["ids"]
    y_true_b = val_data["stream_b"]["y"]
    ids_b = val_data["stream_b"]["ids"]

    df_true_a = pd.DataFrame({"contact_id": ids_a, "target": y_true_a})
    df_true_b = pd.DataFrame({"contact_id": ids_b, "target": y_true_b})
    df_true = pd.concat([df_true_a, df_true_b], axis=0)

    # Merge predictions with ground truth
    df_eval = pd.merge(val_preds, df_true, on="contact_id", how="inner")

    if len(df_eval) == 0:
        print(
            "Error: No overlapping contact_ids between validation predictions and ground truth."
        )
        return

    # Compute MCC
    final_mcc = compute_mcc(df_eval["target"], df_eval["contact"])
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude (Binary classification: 0 or 1)
    df_eval["error"] = np.abs(df_eval["target"] - df_eval["contact"])

    # Prepare features for analysis
    # We concatenate features from both streams. Columns not present in a stream will be NaN.
    X_a = val_data["stream_a"]["X"].copy()
    X_a["contact_id"] = ids_a

    X_b = val_data["stream_b"]["X"].copy()
    X_b["contact_id"] = ids_b

    X_all = pd.concat([X_a, X_b], axis=0, ignore_index=True)

    # Merge errors with features
    df_analysis = pd.merge(
        df_eval[["contact_id", "error"]], X_all, on="contact_id", how="inner"
    )

    # Compute correlations between features and error
    feature_cols = [c for c in df_analysis.columns if c not in ["contact_id", "error"]]
    correlations = df_analysis[feature_cols].corrwith(df_analysis["error"])

    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("Top 10 Feature Correlations with Error:")
    print(correlations_abs.head(10))

    # Print signed correlations for context
    top_10_features = correlations_abs.head(10).index
    print("\nSigned correlations for top features:")
    print(correlations[top_10_features])

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.6804

    if final_mcc > SUBMISSION_THRESHOLD:
        print(
            f"Validation MCC ({final_mcc}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_data, SUBMISSION_PATH)
    else:
        print(
            f"Validation MCC ({final_mcc}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
