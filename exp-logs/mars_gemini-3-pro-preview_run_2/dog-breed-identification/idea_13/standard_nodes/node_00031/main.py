import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image
import torch

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, save_submission
from library.processing import process_stream
from library.training import train_classifier, optimize_ensemble_weights
from library.data import get_class_names


def calculate_sample_log_loss(y_true, y_pred_probs):
    """
    Calculates log loss for each sample individually.
    y_true: (N,) indices
    y_pred_probs: (N, C) probabilities
    """
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)

    # Gather probability of the true class
    n_samples = len(y_true)
    true_class_probs = y_pred_probs[np.arange(n_samples), y_true]

    # Log loss is negative log likelihood
    sample_losses = -np.log(true_class_probs)
    return sample_losses


def perform_failure_analysis(val_df, y_val, val_probs, stream_a_emb, stream_b_emb):
    print("\nPerforming Failure Analysis...")

    # Calculate error per sample
    errors = calculate_sample_log_loss(y_val, val_probs)

    # 1. Feature Norm Correlations (Signal Strength)
    # Norm of the feature vector often correlates with model confidence/image quality
    norm_a = np.linalg.norm(stream_a_emb, axis=1)
    norm_b = np.linalg.norm(stream_b_emb, axis=1)

    corr_a, _ = pearsonr(errors, norm_a)
    corr_b, _ = pearsonr(errors, norm_b)

    print(f"  Correlation (Error vs Stream A Feature Norm): {corr_a:.4f}")
    print(f"  Correlation (Error vs Stream B Feature Norm): {corr_b:.4f}")

    # 2. Image Metadata Correlations (Width, Height, Aspect Ratio)
    # We need to read image dimensions from disk
    widths = []
    heights = []

    # val_df has 'file_path' relative to input dir
    # We iterate and read just the header
    input_dir = Config.INPUT_DIR

    # Limit to first 500 for speed if dataset is huge, but 1840 is fine
    print("  Reading image dimensions for correlation analysis...")
    for idx, row in val_df.iterrows():
        full_path = os.path.join(input_dir, row["file_path"])
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except:
            widths.append(0)
            heights.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.divide(
        widths, heights, out=np.zeros_like(widths, dtype=float), where=heights != 0
    )

    corr_w, _ = pearsonr(errors, widths)
    corr_h, _ = pearsonr(errors, heights)
    corr_ar, _ = pearsonr(errors, aspect_ratios)

    print(f"  Correlation (Error vs Image Width): {corr_w:.4f}")
    print(f"  Correlation (Error vs Image Height): {corr_h:.4f}")
    print(f"  Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")


def main():
    seed_everything(Config.SEED)

    print("=== Starting Hybrid Supervised-SSL Multi-View Ensemble Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Feature Extraction (or Loading Cache)
    # -------------------------------------------------------------------------
    # Stream A: ConvNeXt Large (Supervised)
    data_a = process_stream("stream_a", load_cached_data=True)
    X_train_a, y_train_a = data_a["train"]
    X_val_a, y_val_a = data_a["val"]
    X_test_a, test_ids_a = data_a["test"]

    # Stream B: DINOv2 ViT Large (Self-Supervised)
    data_b = process_stream("stream_b", load_cached_data=True)
    X_train_b, y_train_b = data_b["train"]
    X_val_b, y_val_b = data_b["val"]
    X_test_b, test_ids_b = data_b["test"]

    # Sanity Checks
    assert np.array_equal(y_train_a, y_train_b), "Train label mismatch"
    assert np.array_equal(y_val_a, y_val_b), "Val label mismatch"
    assert np.array_equal(test_ids_a, test_ids_b), "Test ID mismatch"

    # -------------------------------------------------------------------------
    # 2. Classifier Training
    # -------------------------------------------------------------------------
    # Train Stream A Head
    model_a, val_probs_a = train_classifier(
        X_train_a,
        y_train_a,
        X_val_a,
        y_val_a,
        stream_name="stream_a",
        load_cached_data=True,
    )

    # Train Stream B Head
    model_b, val_probs_b = train_classifier(
        X_train_b,
        y_train_b,
        X_val_b,
        y_val_b,
        stream_name="stream_b",
        load_cached_data=True,
    )

    # -------------------------------------------------------------------------
    # 3. Ensemble Optimization
    # -------------------------------------------------------------------------
    # Optimize weights on Validation set
    w_a, w_b = optimize_ensemble_weights(val_probs_a, val_probs_b, y_val_a)

    # Compute Final Validation Predictions
    final_val_probs = (w_a * val_probs_a) + (w_b * val_probs_b)
    final_val_probs /= final_val_probs.sum(axis=1, keepdims=True)

    # Compute Metric
    final_metric = log_loss(y_val_a, final_val_probs)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    # Load validation metadata to link back to images
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    perform_failure_analysis(
        val_df=val_df,
        y_val=y_val_a,
        val_probs=final_val_probs,
        stream_a_emb=X_val_a,
        stream_b_emb=X_val_b,
    )

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold check as per instructions
    TARGET_THRESHOLD = 0.11640673500383826

    if final_metric < TARGET_THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({TARGET_THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        print("  Predicting Stream A...")
        test_probs_a = model_a.predict_proba(X_test_a)

        print("  Predicting Stream B...")
        test_probs_b = model_b.predict_proba(X_test_b)

        # Weighted Ensemble
        final_test_probs = (w_a * test_probs_a) + (w_b * test_probs_b)
        final_test_probs /= final_test_probs.sum(axis=1, keepdims=True)

        # Get class names
        class_names = get_class_names()

        # Save
        save_submission(test_ids_a, final_test_probs, class_names)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({TARGET_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
