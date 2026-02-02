import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from the provided library
from library.config import (
    SEED,
    INPUT_DIR,
    VAL_METADATA_PATH,
    STREAMS,
    NUM_CLASSES,
    WORKING_DIR,
)
from library.training import train_stream
from library.ensemble import optimize_ensemble_weights, generate_submission


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(val_ids, val_y, val_probs, class_labels):
    """
    Analyzes model failures by correlating error magnitude with image metadata.
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate per-sample Log Loss (Error Magnitude)
    # We clip probabilities to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Extract the probability assigned to the true class
    # val_y is (N,), val_probs is (N, C)
    rows = np.arange(len(val_y))
    true_class_probs = val_probs_clipped[rows, val_y]

    # Log loss per sample = -log(p_true)
    sample_errors = -np.log(true_class_probs)

    # 2. Load Metadata to get file paths
    val_df = pd.read_csv(VAL_METADATA_PATH)
    # Map IDs to file paths
    id_to_path = dict(zip(val_df["id"], val_df["file_path"]))

    # 3. Extract Image Features (Width, Height, Aspect Ratio)
    widths = []
    heights = []
    aspect_ratios = []

    # Ensure order matches val_ids
    print("Extracting metadata features for failure analysis...")
    for img_id in val_ids:
        rel_path = id_to_path.get(img_id)
        if not rel_path:
            # Fallback if ID not found (should not happen)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            continue

        full_path = os.path.join(INPUT_DIR, rel_path)
        # Read image dimensions only (don't load pixel data to save memory)
        # Using PIL might be faster for just size, but cv2 is standard here
        img = cv2.imread(full_path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # 4. Compute Correlations
    # Filter out invalid images (0 dims)
    valid_mask = widths > 0

    if np.sum(valid_mask) > 10:
        corr_w, _ = pearsonr(sample_errors[valid_mask], widths[valid_mask])
        corr_h, _ = pearsonr(sample_errors[valid_mask], heights[valid_mask])
        corr_ar, _ = pearsonr(sample_errors[valid_mask], aspect_ratios[valid_mask])

        print(f"Correlation between Error and Width:        {corr_w:.4f}")
        print(f"Correlation between Error and Height:       {corr_h:.4f}")
        print(f"Correlation between Error and Aspect Ratio: {corr_ar:.4f}")

        # Interpretation
        if abs(corr_w) > 0.1 or abs(corr_h) > 0.1:
            print(
                ">> Observation: Error magnitude shows some correlation with image size."
            )
        else:
            print(
                ">> Observation: Error magnitude is largely independent of image size."
            )
    else:
        print("Not enough valid images for correlation analysis.")


def main():
    set_seed(SEED)

    # --- 1. Train / Inference on Streams ---
    # Stream A: ConvNeXt Large
    print("Running Stream A (ConvNeXt)...")
    res_a = train_stream(STREAMS["stream_a"], load_cached_model=True)

    # Stream B: MaxViT Large
    print("\nRunning Stream B (MaxViT)...")
    res_b = train_stream(STREAMS["stream_b"], load_cached_model=True)

    # --- 2. Ensemble Optimization ---
    # Ensure alignment
    if not np.array_equal(res_a["val_y"], res_b["val_y"]):
        raise ValueError("Validation labels mismatch between streams!")

    val_y = res_a["val_y"]
    val_ids = res_a["val_ids"]

    # Optimize weights
    w_a, w_b = optimize_ensemble_weights(res_a["val_probs"], res_b["val_probs"], val_y)

    # Calculate Final Validation Predictions
    final_val_probs = w_a * res_a["val_probs"] + w_b * res_b["val_probs"]

    # Compute Metric
    # Explicitly providing labels ensures correct handling if a class is missing in val (unlikely but safe)
    final_metric = log_loss(val_y, final_val_probs, labels=np.arange(NUM_CLASSES))

    print(f"\nFinal Validation Metric: {final_metric}")

    # --- 3. Failure Analysis ---
    perform_failure_analysis(val_ids, val_y, final_val_probs, np.arange(NUM_CLASSES))

    # --- 4. Submission ---
    THRESHOLD = 0.11640673500383826

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensure test IDs match
        if not np.array_equal(res_a["test_ids"], res_b["test_ids"]):
            raise ValueError("Test IDs mismatch between streams!")

        generate_submission(
            res_a["test_probs"], res_b["test_probs"], w_a, w_b, res_a["test_ids"]
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not improve over threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
