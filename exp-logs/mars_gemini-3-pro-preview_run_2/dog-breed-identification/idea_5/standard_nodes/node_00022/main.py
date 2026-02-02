import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from PIL import Image
import torch
import random

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

import library.config as config
import library.feature_engine as feature_engine
import library.classifier as classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.SEED)

    # ==========================================
    # 1. Feature Extraction
    # ==========================================
    print("--- Starting Feature Extraction ---")

    # Stream A: ConvNeXt
    print(f"Extracting Stream A ({config.MODEL_A_NAME})...")
    X_train_a, y_train_a, _ = feature_engine.extract_features(
        "train", config.MODEL_A_NAME, config.MODEL_A_WEIGHTS, load_cached_data=True
    )
    X_val_a, y_val_a, _ = feature_engine.extract_features(
        "val", config.MODEL_A_NAME, config.MODEL_A_WEIGHTS, load_cached_data=True
    )

    # Stream B: ViT
    print(f"Extracting Stream B ({config.MODEL_B_NAME})...")
    X_train_b, y_train_b, _ = feature_engine.extract_features(
        "train", config.MODEL_B_NAME, config.MODEL_B_WEIGHTS, load_cached_data=True
    )
    X_val_b, y_val_b, _ = feature_engine.extract_features(
        "val", config.MODEL_B_NAME, config.MODEL_B_WEIGHTS, load_cached_data=True
    )

    # Verify label consistency
    if not np.array_equal(y_val_a, y_val_b):
        raise ValueError("Validation labels mismatch between Stream A and Stream B.")

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n--- Starting Model Training ---")

    # Train Classifier A
    model_a, probs_val_a = classifier.train_logreg(
        X_train_a, y_train_a, X_val_a, y_val_a, "Stream_A_ConvNeXt"
    )

    # Train Classifier B
    model_b, probs_val_b = classifier.train_logreg(
        X_train_b, y_train_b, X_val_b, y_val_b, "Stream_B_ViT"
    )

    # ==========================================
    # 3. Ensemble Optimization
    # ==========================================
    print("\n--- Optimizing Ensemble ---")

    # Optimize weights
    # Note: We use model_a.classes_ as the reference for column ordering
    best_weight_a = classifier.optimize_ensemble_weights(
        probs_val_a, probs_val_b, y_val_a, labels=model_a.classes_
    )
    weight_b = 1.0 - best_weight_a

    # Calculate Final Ensemble Probabilities
    probs_ensemble = (best_weight_a * probs_val_a) + (weight_b * probs_val_b)

    # Calculate Final Metric
    # Filter y_val to ensure we only evaluate on known classes (though usually all are known)
    known_classes = set(model_a.classes_)
    mask = np.array([y in known_classes for y in y_val_a])

    if mask.sum() > 0:
        final_metric = log_loss(
            y_val_a[mask], probs_ensemble[mask], labels=model_a.classes_
        )
    else:
        final_metric = float("inf")

    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Calculate per-sample loss
    val_losses = []
    valid_indices = []

    # Map class labels to column indices in the probability matrix
    class_to_col = {cls: i for i, cls in enumerate(model_a.classes_)}

    for i in range(len(y_val_a)):
        true_label = y_val_a[i]
        if true_label in class_to_col:
            col_idx = class_to_col[true_label]
            prob = probs_ensemble[i, col_idx]
            # Clip to avoid log(0)
            prob = max(min(prob, 1 - 1e-15), 1e-15)
            loss = -np.log(prob)
            val_losses.append(loss)
            valid_indices.append(i)

    # Load validation metadata to get image paths
    val_df = pd.read_csv(config.VAL_CSV)
    val_df_analyzed = val_df.iloc[valid_indices].copy()
    val_df_analyzed["loss"] = val_losses

    # Extract image metadata (Width, Height, Aspect Ratio)
    widths = []
    heights = []
    ratios = []

    print("Analyzing image metadata for correlations...")
    for idx, row in val_df_analyzed.iterrows():
        img_path = os.path.join(config.INPUT_DIR, row["file_path"])
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
        except Exception as e:
            # Fallback for read errors
            widths.append(0)
            heights.append(0)
            ratios.append(0)

    val_df_analyzed["width"] = widths
    val_df_analyzed["height"] = heights
    val_df_analyzed["aspect_ratio"] = ratios

    # Calculate Correlations
    corr_width = val_df_analyzed["loss"].corr(val_df_analyzed["width"])
    corr_height = val_df_analyzed["loss"].corr(val_df_analyzed["height"])
    corr_ratio = val_df_analyzed["loss"].corr(val_df_analyzed["aspect_ratio"])

    print(f"Correlation between Error and Width: {corr_width}")
    print(f"Correlation between Error and Height: {corr_height}")
    print(f"Correlation between Error and Aspect Ratio: {corr_ratio}")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.11640673500383826

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        classifier.generate_submission(model, debug=False, load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
