import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import from library
from library.config import config
from library.utils import set_seed
from library.model_finetuning import FineTuner
from library.feature_engineering import FeatureExtractor
from library.model_regressor import RankRegressor
from library.metrics import compute_kendall_tau


def reconstruct_order(df_features):
    """
    Reconstructs the full cell order (Code + Markdown) based on predicted ranks.

    Args:
        df_features (pd.DataFrame): DataFrame containing 'id', 'cell_id', 'n_code', and 'pred'.

    Returns:
        dict: {notebook_id: "cell_id1 cell_id2 ..."}
    """
    # Load metadata to locate files for code cell extraction
    # We check both validation and test metadata
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # Create a lookup for file paths
    path_lookup = (
        pd.concat([val_meta[["id", "file_path"]], test_meta[["id", "file_path"]]])
        .set_index("id")["file_path"]
        .to_dict()
    )

    results = {}

    # Group predictions by notebook ID
    preds_map = df_features.groupby("id")

    for notebook_id, group in preds_map:
        if notebook_id not in path_lookup:
            continue

        file_path = os.path.join(config.INPUT_DIR, path_lookup[notebook_id])

        # Read notebook to get code cells (anchors)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        cell_types = data.get("cell_type", {})

        # Extract code cells.
        # Assumption: In the provided JSONs, code cells are in correct relative order.
        # We filter keys by type 'code'.
        code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]

        # List of tuples: (position_index, cell_id)
        ranked_cells = []

        # 1. Place Code Cells at integer indices: 0.0, 1.0, 2.0, ...
        for i, cid in enumerate(code_cells):
            ranked_cells.append((float(i), cid))

        # 2. Place Markdown Cells based on predicted rank
        # 'pred' is normalized [0, 1].
        # Relative Rank = pred * n_code.
        # This places the markdown cell relative to the code cell indices.
        for _, row in group.iterrows():
            cid = row["cell_id"]
            # Ensure n_code is consistent with what we just read
            n_code = len(code_cells)
            if n_code == 0:
                # If no code cells, put at end or start.
                # Model likely predicts 0.5 or similar.
                rank = row["pred"]
            else:
                rank = row["pred"] * n_code

            ranked_cells.append((rank, cid))

        # 3. Sort all cells by position index
        # Stable sort ensures deterministic order for ties
        ranked_cells.sort(key=lambda x: x[0])

        # 4. Generate space-delimited string
        final_order = " ".join([x[1] for x in ranked_cells])
        results[notebook_id] = final_order

    return results


def main():
    # 0. Setup
    set_seed(config.SEED)

    # Regenerate test metadata from sample_submission.csv to ensure we predict on runtime IDs
    print("Regenerating test metadata from sample_submission.csv...")
    sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        df_sample = pd.read_csv(sample_sub_path)
        df_test_meta = pd.DataFrame(
            {"id": df_sample["id"], "file_path": "test/" + df_sample["id"] + ".json"}
        )
        df_test_meta.to_csv(config.TEST_METADATA_PATH, index=False)

    # Enforce Fast Baseline constraints
    # We sample 10,000 notebooks for training/validation to ensure execution < 2 hours
    config.DEBUG_SAMPLE_SIZE = 10000
    print(f"Running pipeline with DEBUG_SAMPLE_SIZE={config.DEBUG_SAMPLE_SIZE}")

    # 1. Contrastive Fine-Tuning
    print("\n=== Step 1: Contrastive Fine-Tuning ===")
    fine_tuner = FineTuner()

    if not os.path.exists(config.FINE_TUNED_MODEL_PATH):
        print("Training semantic backbone...")
        fine_tuner.train(load_cached_data=True)
    else:
        print("Fine-tuned model found. Skipping training.")

    # 2. Feature Extraction
    print("\n=== Step 2: Feature Extraction ===")
    extractor = FeatureExtractor()

    # Extract Train Features
    train_features = extractor.extract_features(
        config.TRAIN_METADATA_PATH,
        config.TRAIN_FEATURES_PATH,
        mode="train",
        load_cached_data=True,
    )

    # Extract Validation Features
    val_features = extractor.extract_features(
        config.VAL_METADATA_PATH,
        config.VAL_FEATURES_PATH,
        mode="val",
        load_cached_data=True,
    )

    # Extract Test Features (Initial pass with sampling)
    test_features = extractor.extract_features(
        config.TEST_METADATA_PATH,
        config.TEST_FEATURES_PATH,
        mode="test",
        load_cached_data=True,
    )

    # 3. Regression Training
    print("\n=== Step 3: Regression Training ===")
    regressor = RankRegressor()

    # Clean Data
    train_features = train_features.dropna(subset=["target"])
    val_features = val_features.dropna(subset=["target"])

    # Train
    regressor.train(train_features, val_features)
    regressor.save(config.LGBM_MODEL_PATH)

    # 4. Validation & Failure Analysis
    print("\n=== Step 4: Validation & Failure Analysis ===")

    # Predict
    val_features["pred"] = regressor.predict(val_features)

    # Reconstruct Orders
    print("Reconstructing validation orders...")
    val_preds_map = reconstruct_order(val_features)

    # Load Ground Truth
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    val_gt_map = dict(zip(val_meta["id"], val_meta["cell_order"]))

    # Filter for common IDs (in case of sampling mismatch)
    common_ids = set(val_preds_map.keys()).intersection(set(val_gt_map.keys()))
    final_preds = {k: val_preds_map[k] for k in common_ids}
    final_gt = {k: val_gt_map[k] for k in common_ids}

    # Compute Metric
    score = compute_kendall_tau(final_preds, final_gt)
    print(f"Final Validation Metric: {score}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    val_features["error"] = np.abs(val_features["target"] - val_features["pred"])

    correlations = {}
    analyze_cols = ["n_code", "md_len", "sim_max", "sim_mean", "best_match_loc"]
    for col in analyze_cols:
        if col in val_features.columns:
            # Drop NaNs for correlation calculation
            valid_df = val_features[[col, "error"]].dropna()
            if len(valid_df) > 1:
                corr, _ = pearsonr(valid_df[col], valid_df["error"])
                correlations[col] = corr
            else:
                correlations[col] = 0.0

    print("Correlation between Error and Features:")
    for k, v in correlations.items():
        print(f"  {k}: {v:.4f}")

    # 5. Submission
    print("\n=== Step 5: Submission ===")

    if score > 0.7633:
        print("Validation score meets threshold. Generating submission...")

        # Check if we need to re-extract full test set
        test_meta_full = pd.read_csv(config.TEST_METADATA_PATH)
        unique_test_ids = test_features["id"].nunique()
        total_test_ids = len(test_meta_full)

        if unique_test_ids < total_test_ids:
            print(
                f"Test features incomplete ({unique_test_ids}/{total_test_ids}). Re-running full test extraction..."
            )

            # Temporarily disable sampling limit
            saved_limit = config.DEBUG_SAMPLE_SIZE
            config.DEBUG_SAMPLE_SIZE = None

            # Re-extract
            test_features = extractor.extract_features(
                config.TEST_METADATA_PATH,
                os.path.join(config.WORKING_DIR, "features_test_full.parquet"),
                mode="test",
                load_cached_data=True,
            )

            # Restore limit
            config.DEBUG_SAMPLE_SIZE = saved_limit

        # Predict on Test
        test_features["pred"] = regressor.predict(test_features)

        # Reconstruct Orders
        test_preds_map = reconstruct_order(test_features)

        # Create Submission File
        submission_rows = []
        for nid in test_meta_full["id"]:
            if nid in test_preds_map:
                submission_rows.append({"id": nid, "cell_order": test_preds_map[nid]})
            else:
                # Fallback: Just list code cells if markdown prediction failed/missing
                # We reuse the logic inside reconstruct_order but for single ID if needed
                # Ideally, test_preds_map should have it.
                submission_rows.append({"id": nid, "cell_order": ""})

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation score {score:.4f} is below threshold 0.7633. Submission skipped."
        )


if __name__ == "__main__":
    main()
