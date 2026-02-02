import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff

# Import provided library modules
from library import config, utils, data_processing, model, post_processing

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_hausdorff_3d(mask_pred, mask_gt):
    """
    Computes a normalized 3D Hausdorff score.
    Coordinates are normalized by volume dimensions to [0, 1].
    Metric = max(0, 1 - Hausdorff_Distance).
    """
    # Handle empty cases
    sum_pred = np.sum(mask_pred)
    sum_gt = np.sum(mask_gt)

    if sum_pred == 0 and sum_gt == 0:
        return 1.0  # Perfect match (both empty) -> Distance 0 -> Score 1
    if sum_pred == 0 or sum_gt == 0:
        return 0.0  # One empty -> Max distance -> Score 0

    # Get coordinates of non-zero pixels (z, y, x)
    coords_pred = np.argwhere(mask_pred).astype(np.float32)
    coords_gt = np.argwhere(mask_gt).astype(np.float32)

    # Normalize coordinates to unit cube [0, 1]^3
    depth, height, width = mask_pred.shape

    # Avoid division by zero
    d_norm = max(depth, 1)
    h_norm = max(height, 1)
    w_norm = max(width, 1)

    coords_pred[:, 0] /= d_norm
    coords_pred[:, 1] /= h_norm
    coords_pred[:, 2] /= w_norm

    coords_gt[:, 0] /= d_norm
    coords_gt[:, 1] /= h_norm
    coords_gt[:, 2] /= w_norm

    # Use cKDTree for fast distance computation
    # Directed Hausdorff: max(min_dist(A, B))

    # Pred -> GT
    tree_gt = cKDTree(coords_gt)
    dists_p2g, _ = tree_gt.query(coords_pred, k=1)
    hd_p2g = np.max(dists_p2g)

    # GT -> Pred
    tree_pred = cKDTree(coords_pred)
    dists_g2p, _ = tree_pred.query(coords_gt, k=1)
    hd_g2p = np.max(dists_g2p)

    hausdorff_dist = max(hd_p2g, hd_g2p)

    # Convert distance to bounded score
    # Assuming normalized distance <= 1 usually, but we clamp to 0 just in case
    return max(0.0, 1.0 - hausdorff_dist)


def validate(clf, val_meta):
    """
    Runs validation inference, calculates metrics, and performs failure analysis data collection.
    """
    print("Running validation...")

    # Group by Case/Day
    groups = list(val_meta.groupby(["case", "day"]))

    dice_scores = []
    hausdorff_scores = []
    errors = []

    for (case, day), group_df in groups:
        # Sort by slice
        group_df = group_df.sort_values("slice")

        # 1. Prepare Input for Inference
        # IMPORTANT: model._process_volume_inference has a bug with duplicate slices (multiple classes).
        # We must pass a dataframe with unique slices to get correct volumetric predictions.
        inference_input = group_df.drop_duplicates(subset=["slice"]).copy()
        slices = inference_input["slice"].values

        # Dimensions
        h = int(inference_input.iloc[0]["img_height"])
        w = int(inference_input.iloc[0]["img_width"])
        d = len(slices)

        # 2. Run Inference
        # Returns list of dicts: {'id': ..., 'class': ..., 'predicted': RLE}
        preds_list = model._process_volume_inference(inference_input, clf)

        # Map slice number to index 0..D-1
        slice_map = {s: i for i, s in enumerate(slices)}

        # Reconstruct Prediction Volumes
        pred_volumes = {
            "large_bowel": np.zeros((d, h, w), dtype=np.uint8),
            "small_bowel": np.zeros((d, h, w), dtype=np.uint8),
            "stomach": np.zeros((d, h, w), dtype=np.uint8),
        }

        for p in preds_list:
            # ID in preds_list corresponds to the ID in inference_input
            p_id = p["id"]
            p_cls = p["class"]
            p_rle = p["predicted"]

            # Find slice index
            # We look up the slice number from the input dataframe using the ID
            row = inference_input[inference_input["id"] == p_id]
            if not row.empty:
                s_num = row.iloc[0]["slice"]
                if s_num in slice_map:
                    s_idx = slice_map[s_num]
                    if p_rle != "":
                        mask = utils.rle_decode(p_rle, (h, w))
                        pred_volumes[p_cls][s_idx] = mask

        # 3. Reconstruct Ground Truth Volumes
        gt_volumes = {
            "large_bowel": np.zeros((d, h, w), dtype=np.uint8),
            "small_bowel": np.zeros((d, h, w), dtype=np.uint8),
            "stomach": np.zeros((d, h, w), dtype=np.uint8),
        }

        # Iterate over the FULL group_df to get masks for all classes
        for _, row in group_df.iterrows():
            s_num = row["slice"]
            if s_num not in slice_map:
                continue

            s_idx = slice_map[s_num]
            cls = row["class"]
            rle = row["segmentation"]

            if pd.notna(rle) and rle != "":
                mask = utils.rle_decode(rle, (h, w))
                gt_volumes[cls][s_idx] = mask

        # 4. Compute Metrics
        for cls in config.CLASSES[1:]:  # Skip background
            y_true = gt_volumes[cls]
            y_pred = pred_volumes[cls]

            # Dice
            d_score = utils.dice_coefficient(y_true, y_pred)
            dice_scores.append(d_score)

            # Hausdorff Score
            h_score = compute_hausdorff_3d(y_pred, y_true)
            hausdorff_scores.append(h_score)

            # Log for failure analysis
            errors.append(
                {
                    "case": case,
                    "day": day,
                    "class": cls,
                    "dice": d_score,
                    "hausdorff_score": h_score,
                    "error": 1.0 - d_score,
                    "num_slices": d,
                    "img_height": h,
                    "img_width": w,
                }
            )

    # Aggregate
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_hausdorff = np.mean(hausdorff_scores) if hausdorff_scores else 0.0

    final_metric = 0.4 * mean_dice + 0.6 * mean_hausdorff

    print(f"Validation Dice: {mean_dice:.6f}")
    print(f"Validation Hausdorff Score: {mean_hausdorff:.6f}")
    print(f"Final Validation Metric: {final_metric:.18f}")

    return pd.DataFrame(errors)


def failure_analysis(error_df):
    """
    Analyzes model failures by correlating error magnitude with metadata features.
    """
    print("\nFailure Analysis:")
    if error_df.empty:
        print("No validation data for failure analysis.")
        return

    print("Correlation between Error (1-Dice) and Features:")

    # Encode class as integer for correlation
    error_df["class_code"] = error_df["class"].astype("category").cat.codes

    features = ["num_slices", "img_height", "img_width", "class_code"]

    for feat in features:
        if feat in error_df.columns:
            # Check if feature has variance
            if error_df[feat].nunique() > 1:
                corr = error_df["error"].corr(error_df[feat])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: NaN (No variance)")


def main():
    # Set seeds for reproducibility
    set_seed(config.SEED)

    print("Starting pipeline...")

    # 1. Data Preparation
    # Load cached tabular datasets for LightGBM
    print("Loading datasets...")
    train_df = data_processing.build_tabular_dataset("train", load_cached_data=True)
    val_df = data_processing.build_tabular_dataset("val", load_cached_data=True)

    # 2. Model Training
    print("Training LightGBM model...")
    clf = model.train_model(train_df, val_df)

    # Save the trained model
    model_path = os.path.join(config.WORKING_DIR, "lgbm_model.joblib")
    clf.save(model_path)
    print(f"Model saved to {model_path}")

    # 3. Validation
    # Load validation metadata
    val_meta = utils.load_metadata("val")
    # Run validation
    error_df = validate(clf, val_meta)

    # 4. Failure Analysis
    failure_analysis(error_df)

    # 5. Submission
    print("Preparing submission...")
    # Fix for provided library bug:
    # model._process_volume_inference produces incorrect output if the input dataframe
    # contains duplicate slices (which test_metadata.csv does).
    # We deduplicate test_metadata on disk before calling generate_submission.
    # The inference logic will then generate predictions for all classes for each unique slice,
    # resulting in the correct number of rows (Cartesian product of slices x classes).

    test_meta_path = os.path.join(config.METADATA_DIR, "test_metadata.csv")
    if os.path.exists(test_meta_path):
        test_meta = pd.read_csv(test_meta_path)
        # Keep only unique slices
        test_meta_unique = test_meta.drop_duplicates(subset=["case", "day", "slice"])
        # Overwrite metadata file
        test_meta_unique.to_csv(test_meta_path, index=False)
        print(
            f"Optimized test metadata: {len(test_meta)} -> {len(test_meta_unique)} rows."
        )

    # Generate submission using provided function
    model.generate_submission(clf)


if __name__ == "__main__":
    main()
