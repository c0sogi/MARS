import sys
import os
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Ensure library imports work
sys.path.append(".")

from library.config import Config
from library.classifier import StreamClassifier


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_per_sample_log_loss(y_true, y_pred, eps=1e-15):
    """Calculates Log Loss for each sample individually."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n_samples = len(y_true)
    # y_true contains class indices
    prob_true = y_pred[np.arange(n_samples), y_true]
    return -np.log(prob_true)


def run_failure_analysis(y_true, y_pred, metadata_path):
    """
    Performs failure analysis by correlating prediction error with image metadata.
    """
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude (Log Loss) per sample
    errors = calculate_per_sample_log_loss(y_true, y_pred)

    # Load metadata
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)

    # Ensure alignment
    if len(df) != len(errors):
        print(
            f"Warning: Metadata length ({len(df)}) does not match predictions ({len(errors)}). Skipping analysis."
        )
        return

    widths = []
    heights = []
    aspect_ratios = []

    print("Collecting image statistics for failure analysis...")
    input_dir = Config.INPUT_DIR

    for _, row in df.iterrows():
        path = os.path.join(input_dir, row["file_path"])
        try:
            # Read image to get dimensions
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
                aspect_ratios.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Filter valid images
    valid_mask = ~np.isnan(widths)

    if np.sum(valid_mask) < 10:
        print("Not enough valid images for correlation analysis.")
        return

    # Calculate Correlations
    # We check if features are constant to avoid warnings
    def safe_pearson(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return pearsonr(x, y)[0]

    corr_w = safe_pearson(errors[valid_mask], widths[valid_mask])
    corr_h = safe_pearson(errors[valid_mask], heights[valid_mask])
    corr_ar = safe_pearson(errors[valid_mask], aspect_ratios[valid_mask])

    print(f"Correlation between Error and Width: {corr_w:.8f}")
    print(f"Correlation between Error and Height: {corr_h:.8f}")
    print(f"Correlation between Error and Aspect Ratio: {corr_ar:.8f}")


def main():
    # 1. Initialization
    set_seed(Config.SEED)
    print("Initializing Dual-Stream Heterogeneous Multi-View Ensemble...")

    classifier = StreamClassifier()

    # 2. Train Stream A (ConvNeXt)
    # Using full dataset (debug_sample_size=None) to ensure metric threshold is met.
    # The feature-extraction approach is fast enough to run fully within the time limit.
    model_a, probs_a, y_val_a = classifier.train_stream(
        "stream_a", debug_sample_size=None
    )

    # 3. Train Stream B (ViT)
    model_b, probs_b, y_val_b = classifier.train_stream(
        "stream_b", debug_sample_size=None
    )

    # Verify label alignment
    if not np.array_equal(y_val_a, y_val_b):
        raise ValueError("Validation labels mismatch between streams.")

    # 4. Optimize Ensemble Weights
    weights_data = classifier.optimize_ensemble(probs_a, probs_b, y_val_a)

    # 5. Report Validation Metric
    final_metric = weights_data["val_loss"]
    # Printing full precision as required
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Reconstruct ensemble predictions for analysis
    w_a = weights_data["w_a"]
    w_b = weights_data["w_b"]
    ensemble_probs = w_a * probs_a + w_b * probs_b

    run_failure_analysis(y_val_a, ensemble_probs, Config.VAL_METADATA_PATH)

    # 7. Conditional Submission
    THRESHOLD = 0.11640673500383826

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        classifier.generate_submission(
            model_a, model_b, weights_data, debug_sample_size=None
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
