import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration Patching
# ==========================================
# We patch the configuration to optimize for speed and stability
# on a tiny debug dataset.
import library.config

library.config.N_FOLDS = 2  # Reduce folds to 2 for speed
library.config.NUM_WORKERS = 0  # Disable multiprocessing for small data overhead

# Import library modules after patching
from library.config import SEED, SUBMISSION_DIR, CACHE_DIR
from library.utils import seed_everything
from library.feature_extraction import run_extraction
from library.data_processor import load_dataset
from library.model_builder import create_pipeline, aggregate_predictions
from library.workflow import train_ensemble, predict_ensemble


def run_demo():
    print("=== Starting Leaf Classification Library Demo ===")

    # Setup
    seed_everything(SEED)
    debug_size = 10  # Process only 10 images per split for speed

    # Clean cache to ensure we demonstrate the computation logic
    if os.path.exists(CACHE_DIR):
        print(f"Cleaning cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # 2. Feature Extraction Demonstration
    # ==========================================
    print(f"\n[Step 1] Running Feature Extraction (Debug Size: {debug_size})")
    # load_cached_data=False forces the extraction pipeline to run
    train_raw, val_raw, test_raw = run_extraction(
        load_cached_data=False, debug_sample_size=debug_size
    )

    # Validation: Check feature dimensions
    # DINOv2 Large: 1024 dim, ConvNeXt Large: 1536 dim
    print("Verifying extraction output shapes...")

    # Check Train
    # Output shape is (N_Images, N_Views, Feature_Dim)
    n_train_imgs = train_raw["dino"].shape[0]
    n_views = train_raw["dino"].shape[1]

    assert (
        n_train_imgs == debug_size
    ), f"Got {n_train_imgs} images, expected {debug_size}"
    assert n_views == 12, f"Got {n_views} views per image, expected 12"
    assert (
        train_raw["dino"].shape[2] == 1024
    ), f"DINO dim mismatch: {train_raw['dino'].shape[2]}"
    assert (
        train_raw["conv"].shape[2] == 1536
    ), f"ConvNeXt dim mismatch: {train_raw['conv'].shape[2]}"
    assert len(train_raw["ids"]) == n_train_imgs, "ID count does not match image count"

    print("Feature extraction verification passed.")

    # ==========================================
    # 3. Data Processing Demonstration
    # ==========================================
    print(f"\n[Step 2] Running Data Processing (Densification & Merging)")
    # This step aggregates views into 3 centroids and merges tabular data
    data = load_dataset(load_cached_data=False, debug_sample_size=debug_size)

    print("Verifying processed dataset...")
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    ids_train = data["train"]["ids"]

    # Calculate expected dimension: DINO(1024) + Conv(1536) + Tabular(192) = 2752
    expected_dim = 1024 + 1536 + 192

    assert (
        X_train.shape[1] == expected_dim
    ), f"Expected feature dim {expected_dim}, got {X_train.shape[1]}"
    assert X_train.shape[0] == len(y_train), "X and y length mismatch"

    # Verify Densification: Should be 3 rows per unique image ID
    unique_ids = np.unique(ids_train)
    assert (
        X_train.shape[0] == len(unique_ids) * 3
    ), "Densification factor mismatch (expected 3x)"

    print("Data processing verification passed.")

    # ==========================================
    # 4. Model Pipeline Demonstration
    # ==========================================
    print(f"\n[Step 3] Building and Testing Model Pipeline")
    feature_indices = data["feature_indices"]
    pipeline = create_pipeline(feature_indices)

    print("Fitting pipeline on debug training data...")
    pipeline.fit(X_train, y_train)

    print("Predicting on validation set...")
    X_val = data["val"]["X"]
    ids_val = data["val"]["ids"]

    # Predict probabilities on densified data
    probs_dense = pipeline.predict_proba(X_val)

    # Aggregate back to image level
    agg_ids, agg_probs = aggregate_predictions(probs_dense, ids_val)

    # Verification
    assert len(agg_ids) == len(np.unique(ids_val)), "Aggregated ID count mismatch"
    assert agg_probs.shape[1] == len(
        data["classes"]
    ), "Probability class count mismatch"
    assert np.all(
        (agg_probs >= 0) & (agg_probs <= 1)
    ), "Probabilities out of range [0, 1]"

    print("Model pipeline verification passed.")

    # ==========================================
    # 5. Full Workflow Demonstration
    # ==========================================
    print(f"\n[Step 4] Running Full Workflow (Ensemble Training & Inference)")

    # Train Ensemble (Runs K-Fold CV)
    # Note: We patched N_FOLDS=2, so this runs quickly.
    print("Executing train_ensemble...")
    avg_log_loss = train_ensemble(debug_sample_size=debug_size, load_cached_data=True)
    print(f"Workflow Training Log Loss: {avg_log_loss:.4f}")

    # Generate Submission
    print("Executing predict_ensemble...")
    predict_ensemble(debug_sample_size=debug_size, load_cached_data=True)

    # Verify Submission File
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission generated. Shape: {df_sub.shape}")

    # Check format: id column + 99 species columns = 100 columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        len(df_sub.columns) == 100
    ), f"Expected 100 columns, got {len(df_sub.columns)}"

    # Check that probabilities roughly sum to 1 (normalized)
    # We check the first row
    row_sum = df_sub.iloc[0, 1:].sum()
    assert np.isclose(row_sum, 1.0), f"Probabilities do not sum to 1, got {row_sum}"

    print("Workflow verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
