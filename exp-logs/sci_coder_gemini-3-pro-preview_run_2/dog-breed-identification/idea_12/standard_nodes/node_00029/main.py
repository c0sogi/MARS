import sys
import os
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.feature_manager import extract_features
from library.linear_probe import StreamClassifier
from library.ensemble_optimizer import (
    optimize_ensemble_weights,
    blend_predictions,
    generate_submission,
)


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_failure_analysis(val_labels, val_preds, classes, train_labels):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and class frequency.
    """
    print("\n--- Failure Analysis ---")

    # Map class names to indices
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Get indices of true labels
    y_true_indices = np.array([class_to_idx[lbl] for lbl in val_labels])

    # Extract probabilities assigned to the true class
    # Clip to avoid log(0)
    eps = 1e-15
    probs_clipped = np.clip(val_preds, eps, 1 - eps)

    # Advanced indexing to get prob of true class for each sample
    true_class_probs = probs_clipped[np.arange(len(y_true_indices)), y_true_indices]

    # Calculate Log Loss per sample (Error Magnitude)
    sample_losses = -np.log(true_class_probs)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame({"label": val_labels, "loss": sample_losses})

    # Calculate Class Frequency in Training Set
    train_class_counts = pd.Series(train_labels).value_counts()
    df_analysis["train_count"] = df_analysis["label"].map(train_class_counts)

    # 1. Correlation between Error and Class Frequency
    # We expect negative correlation (more samples -> lower loss)
    corr_freq = df_analysis["loss"].corr(df_analysis["train_count"])
    print(f"Correlation between Error Magnitude and Class Frequency: {corr_freq}")

    # 2. Identify Hardest Classes
    class_avg_loss = (
        df_analysis.groupby("label")["loss"].mean().sort_values(ascending=False)
    )
    print("\nTop 5 Hardest Classes (Highest Avg Log Loss):")
    print(class_avg_loss.head(5))

    return corr_freq


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Starting execution on device: {Config.DEVICE}")

    # 2. Feature Extraction (Train & Val)
    # We use load_cached_data=True to utilize any existing work,
    # but since this is a fresh run in a new env, it will likely compute.

    print("\n--- Processing Stream A (ConvNeXt) ---")
    train_emb_a, train_ids_a, train_lbl_a = extract_features(Config.STREAM_A, "train")
    val_emb_a, val_ids_a, val_lbl_a = extract_features(Config.STREAM_A, "val")

    print("\n--- Processing Stream B (EVA02) ---")
    train_emb_b, train_ids_b, train_lbl_b = extract_features(Config.STREAM_B, "train")
    val_emb_b, val_ids_b, val_lbl_b = extract_features(Config.STREAM_B, "val")

    # Verify alignment
    if not np.array_equal(train_lbl_a, train_lbl_b):
        raise ValueError("Training labels mismatch between Stream A and Stream B.")
    if not np.array_equal(val_lbl_a, val_lbl_b):
        raise ValueError("Validation labels mismatch between Stream A and Stream B.")

    # 3. Train Classifiers
    print("\n--- Training Classifiers ---")

    # Stream A
    clf_a = StreamClassifier(Config.STREAM_A)
    clf_a.train(train_emb_a, train_lbl_a)

    # Stream B
    clf_b = StreamClassifier(Config.STREAM_B)
    clf_b.train(train_emb_b, train_lbl_b)

    # 4. Validation & Ensemble Optimization
    print("\n--- Validation & Optimization ---")

    # Predict on Validation set
    val_preds_a = clf_a.predict(val_emb_a)
    val_preds_b = clf_b.predict(val_emb_b)

    # Get class names (should be identical)
    classes = clf_a.classes_

    # Optimize weights
    best_w_a = optimize_ensemble_weights(val_preds_a, val_preds_b, val_lbl_a, classes)

    # Blend Predictions
    final_val_preds = blend_predictions(val_preds_a, val_preds_b, best_w_a)

    # Calculate Final Metric
    final_metric = log_loss(val_lbl_a, final_val_preds, labels=classes)

    # PRINT REQUIRED METRIC EXACTLY AS REQUESTED
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    calculate_failure_analysis(val_lbl_a, final_val_preds, classes, train_lbl_a)

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.11640673500383826

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Extract Test Features
        print("\n--- Processing Test Set ---")
        test_emb_a, test_ids_a, _ = extract_features(Config.STREAM_A, "test")
        test_emb_b, test_ids_b, _ = extract_features(Config.STREAM_B, "test")

        # Verify alignment
        if not np.array_equal(test_ids_a, test_ids_b):
            raise ValueError("Test IDs mismatch between Stream A and Stream B.")

        # Predict
        test_preds_a = clf_a.predict(test_emb_a)
        test_preds_b = clf_b.predict(test_emb_b)

        # Blend
        final_test_preds = blend_predictions(test_preds_a, test_preds_b, best_w_a)

        # Generate Submission File
        generate_submission(test_ids_a, final_test_preds, classes)

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
