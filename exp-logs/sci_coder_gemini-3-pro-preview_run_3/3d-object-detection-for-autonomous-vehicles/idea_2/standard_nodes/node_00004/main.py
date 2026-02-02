import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.dataset_factory import build_tabular_dataset
from library.lgbm_model import ObjectDetector
from library.utils import calc_iou_3d
from library.feature_engineering import parse_label_string

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    Config.set_seed(seed)


def calculate_custom_metric(val_preds_df, val_metadata_df, iou_thresholds):
    """
    Calculates the Mean Average Precision at different IoU thresholds as defined in the task.
    Metric: Mean over thresholds of (Mean over images of (TP / (TP + FP + FN)))
    """
    # 1. Prepare Ground Truth Data
    # Map sample_token -> list of GT boxes
    gt_map = {}
    for _, row in val_metadata_df.iterrows():
        token = row["sample_token"]
        label_str = row.get("label", "")
        gt_objects = parse_label_string(label_str)
        # Store as list of dicts: {'box': np.array, 'class_name': str}
        gt_map[token] = gt_objects

    # 2. Prepare Prediction Data
    # Map sample_token -> list of Pred boxes (sorted by confidence)
    pred_map = {token: [] for token in gt_map.keys()}

    if not val_preds_df.empty:
        # Sort by confidence descending
        val_preds_df = val_preds_df.sort_values(by="confidence", ascending=False)

        for _, row in val_preds_df.iterrows():
            token = row["sample_token"]
            if token not in pred_map:
                continue  # Should not happen if val_metadata matches val_preds scope

            # Construct box: x, y, z, w, l, h, yaw
            box = np.array(
                [
                    row["final_x"],
                    row["final_y"],
                    row["final_z"],
                    row["final_w"],
                    row["final_l"],
                    row["final_h"],
                    row["final_yaw"],
                ]
            )
            pred_map[token].append(
                {
                    "box": box,
                    "class_name": row["class_name"],
                    "confidence": row["confidence"],
                }
            )

    # 3. Calculate Metric
    # Outer loop: Thresholds
    threshold_scores = []

    for t in iou_thresholds:
        image_scores = []

        # Inner loop: Images
        for token, gt_objs in gt_map.items():
            preds = pred_map[token]

            # If no GT and no Preds, score is 0 or 1?
            # Task: "If there are no ground truth objects... ANY number of predictions... score of zero"
            # Implies if GT is empty:
            #   If Preds > 0 -> Score 0
            #   If Preds == 0 -> TP=0, FP=0, FN=0 -> 0/0? Usually 1.0 for perfect empty match, but let's follow formula.
            #   Formula: TP / (TP + FP + FN). If all 0, undefined.
            #   However, usually in detection, empty image with no detections is perfect.
            #   Let's strictly follow the formula. If denominator is 0, define as 1.0 (perfect match of nothingness).

            tp = 0
            fp = 0
            fn = 0

            # Matching logic
            # We need to match preds to GTs. Each GT can be matched at most once.
            # Preds are already sorted by confidence.

            matched_gt_indices = set()

            for p in preds:
                # Find best matching GT
                best_iou = -1.0
                best_gt_idx = -1

                for idx, g in enumerate(gt_objs):
                    if idx in matched_gt_indices:
                        continue

                    # Check class match (Task doesn't explicitly say class match is required for IoU,
                    # but usually it is for detection. The prompt says "predicted object matches a ground truth object".
                    # Usually implies class consistency. Let's assume class must match.)
                    if p["class_name"] != g["class_name"]:
                        continue

                    iou = calc_iou_3d(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > t:
                    tp += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    fp += 1

            # FN are GTs that were not matched
            fn = len(gt_objs) - len(matched_gt_indices)

            # Calculate Precision for this image at threshold t
            denominator = tp + fp + fn
            if denominator == 0:
                score = 1.0  # Both empty
            else:
                score = tp / denominator

            image_scores.append(score)

        # Average over images
        avg_score_t = np.mean(image_scores)
        threshold_scores.append(avg_score_t)

    # Final mAP is mean over thresholds
    final_metric = np.mean(threshold_scores)
    return final_metric


def perform_failure_analysis(detector, val_df):
    """
    Analyzes correlations between errors and features on the validation set.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Filter for positive samples to analyze regression error
    # We use the raw features and targets from val_df
    pos_mask = val_df["target_class"] > 0
    if pos_mask.sum() == 0:
        print("No positive samples in validation set for analysis.")
        return

    pos_df = val_df[pos_mask].copy()

    # Get predictions for residuals
    X_pos = pos_df[Config.FEATURES]

    # Calculate errors for each regression target
    # We want to see how error correlates with 'num_points' and distance

    # Calculate distance from sensor (0,0,0)
    pos_df["dist_to_sensor"] = np.sqrt(
        pos_df["center_x"] ** 2 + pos_df["center_y"] ** 2
    )

    error_dict = {}

    # We'll compute a composite error metric: Mean Absolute Error across all spatial targets
    total_abs_error = np.zeros(len(pos_df))

    spatial_targets = ["dx", "dy", "dz"]

    for target in spatial_targets:
        if target in detector.regressors:
            model = detector.regressors[target]
            preds = model.predict(X_pos)
            actuals = pos_df[target].values
            abs_err = np.abs(preds - actuals)
            total_abs_error += abs_err

    pos_df["total_spatial_error"] = total_abs_error

    # Correlations
    features_to_check = [
        "num_points",
        "dist_to_sensor",
        "intensity_mean",
        "sphericity",
        "planarity",
    ]

    print(f"{'Feature':<20} | {'Correlation with Spatial Error':<30} | {'P-Value':<10}")
    print("-" * 70)

    for feat in features_to_check:
        if feat in pos_df.columns:
            # Drop NaNs if any
            valid_data = pos_df[[feat, "total_spatial_error"]].dropna()
            if len(valid_data) > 1:
                corr, p_val = pearsonr(
                    valid_data[feat], valid_data["total_spatial_error"]
                )
                print(f"{feat:<20} | {corr:<30.4f} | {p_val:<10.4f}")


def main():
    set_seed(Config.RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 1. Data Loading & Preparation
    # --------------------------------------------------------------------------
    # We limit samples for Train/Val to ensure the baseline runs quickly (within 2h).
    # Test set is processed in full.

    print("Building Training Dataset...")
    train_df = build_tabular_dataset("train", max_samples=5000, load_cached_data=False)

    print("Building Validation Dataset...")
    val_df = build_tabular_dataset("val", max_samples=1000, load_cached_data=False)

    print("Building Test Dataset...")
    test_df = build_tabular_dataset("test", load_cached_data=False)  # Full test set

    # --------------------------------------------------------------------------
    # 2. Model Training
    # --------------------------------------------------------------------------
    detector = ObjectDetector()
    detector.train(train_df, val_df)

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("Running Validation Inference...")
    val_preds = detector.predict(val_df)

    # Load metadata to get Ground Truth
    # Since we used max_samples, build_tabular_dataset created a subset metadata file.
    # However, we can simply filter the full metadata or the subset if we knew the name.
    # The safest way is to load the subset file generated by dataset_factory.
    # Based on logic: f"{mode}_metadata_subset_{max_samples}.csv"
    val_meta_path = os.path.join(Config.WORKING_DIR, f"val_metadata_subset_1000.csv")
    if not os.path.exists(val_meta_path):
        # Fallback to full metadata if subset file not found (e.g. if cache was hit but file deleted)
        val_meta_path = Config.VAL_METADATA_PATH

    val_meta_df = pd.read_csv(val_meta_path)

    # Filter metadata to only include samples present in our val_df (in case of mismatch)
    valid_tokens = set(val_df["sample_token"].unique())
    val_meta_df = val_meta_df[val_meta_df["sample_token"].isin(valid_tokens)]

    print("Calculating Metric...")
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    final_metric = calculate_custom_metric(val_preds, val_meta_df, iou_thresholds)

    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    perform_failure_analysis(detector, val_df)

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    detector.generate_submission(test_df)
    print("Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
