import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr

import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_extractor as feature_extractor
import library.ensemble_logic as ensemble_logic

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)

    # 2. Load Model (Backbone)
    # We load the backbone once and reuse it across all views to save time and memory.
    # The model handles variable input sizes (AdaptiveAvgPool), so we don't need to reload it.
    print("Loading backbone model...")
    model = feature_extractor.load_backbone(config.DEVICE)

    views = ["standard", "global", "local"]
    view_data = {}

    # 3. Feature Extraction Phase
    # Iterate through each view to extract embeddings for Train, Val, and Test sets.
    print("Starting feature extraction...")
    for view in views:
        # Create DataLoaders for the specific view (handling resizing/cropping)
        train_loader, val_loader, test_loader, classes = data_loader.create_loaders(
            view
        )

        # Extract Embeddings (uses caching if available)
        # We pass the loaded model to avoid reloading weights
        train_emb, train_lbl = feature_extractor.get_embeddings(
            view, "train", train_loader, load_cached_data=True, model=model
        )
        val_emb, val_lbl = feature_extractor.get_embeddings(
            view, "val", val_loader, load_cached_data=True, model=model
        )
        test_emb, test_ids = feature_extractor.get_embeddings(
            view, "test", test_loader, load_cached_data=True, model=model
        )

        # Store data for this view
        view_data[view] = {
            "train_X": train_emb,
            "train_y": train_lbl,
            "val_X": val_emb,
            "val_y": val_lbl,
            "test_X": test_emb,
            "test_ids": test_ids,
            "classes": classes,
        }

    # Free GPU memory as we only need embeddings for the next steps
    del model
    torch.cuda.empty_cache()

    # 4. Training Phase (Early Fusion)
    # Concatenate embeddings from all views and train a single robust classifier.
    # Cite Lesson 00009: Testing Early Fusion on homogeneous backbones.
    print("Preparing Early Fusion features...")

    # Concatenate Train Features
    # Note: We ensured shuffle=False in data_loader, so rows align.
    train_X_all = np.hstack([view_data[v]["train_X"] for v in views])
    train_y = view_data[views[0]]["train_y"]  # Labels are same

    # Concatenate Val Features
    val_X_all = np.hstack([view_data[v]["val_X"] for v in views])
    y_val_true = view_data[views[0]]["val_y"].astype(int)

    # Concatenate Test Features
    test_X_all = np.hstack([view_data[v]["test_X"] for v in views])

    # Shuffle training data before fitting (since loader was non-shuffled)
    from sklearn.utils import shuffle

    train_X_all, train_y = shuffle(train_X_all, train_y, random_state=config.SEED)

    print(f"Combined Feature Shape: {train_X_all.shape}")

    # Train with CV for hyperparameter tuning
    clf, final_val_probs, final_metric = ensemble_logic.train_logreg_cv(
        train_X_all, train_y, val_X_all, y_val_true, name="early_fusion_multiview"
    )

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Correlate error magnitude with image properties (Width, Height, Aspect Ratio)
    print("Performing Failure Analysis...")

    # Calculate per-sample Log Loss
    # Select the probability assigned to the true class
    rows = np.arange(len(y_val_true))
    true_class_probs = final_val_probs[rows, y_val_true]
    # Clip to avoid log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_class_probs)

    # Extract image metadata for validation set
    val_df = pd.read_csv(config.VAL_CSV)

    widths = []
    heights = []

    # Efficiently read image dimensions
    for idx, row in val_df.iterrows():
        img_path = os.path.join(config.INPUT_DIR, row["file_path"])
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)

    widths = np.array(widths, dtype=float)
    heights = np.array(heights, dtype=float)

    # Calculate Aspect Ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratios = widths / heights

    # Compute Correlations (ignoring NaNs)
    valid_mask = ~np.isnan(widths)

    if np.sum(valid_mask) > 1:
        corr_w, _ = pearsonr(sample_losses[valid_mask], widths[valid_mask])
        corr_h, _ = pearsonr(sample_losses[valid_mask], heights[valid_mask])
        corr_ar, _ = pearsonr(sample_losses[valid_mask], aspect_ratios[valid_mask])

        print(f"Correlation between Error and Image Width: {corr_w:.4f}")
        print(f"Correlation between Error and Image Height: {corr_h:.4f}")
        print(f"Correlation between Error and Aspect Ratio: {corr_ar:.4f}")
    else:
        print("Insufficient valid metadata for correlation analysis.")

    # 8. Submission Generation
    # Generate submission only if metric is better than threshold
    THRESHOLD = 0.11882943387452517

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Predict on Test Set
        final_test_probs = clf.predict_proba(test_X_all)

        # Prepare Submission DataFrame
        # Get IDs from one of the views
        test_ids = view_data["standard"]["test_ids"]
        # Get Class Names
        classes = view_data["standard"]["classes"]

        sub_df = pd.DataFrame(final_test_probs, columns=classes)
        sub_df.insert(0, "id", test_ids)

        # Save
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
