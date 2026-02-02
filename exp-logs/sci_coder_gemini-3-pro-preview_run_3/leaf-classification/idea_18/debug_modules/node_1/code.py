import os
import sys
import shutil
import numpy as np
import pandas as pd
import joblib
import torch

# ==========================================
# 0. Patch TQDM to be silent
# ==========================================
# We must patch this before importing library modules that use tqdm
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# ==========================================
# 1. Import Library Modules
# ==========================================
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.feature_extractor import FeatureExtractor, get_rotated_views
from library.data_manager import DataManager
from library.model_factory import ModelFactory
from library.workflow import Workflow


def main():
    # ==========================================
    # 2. Configuration & Setup
    # ==========================================
    print("--- Setting up Demonstration Environment ---")

    # Override Config for rapid execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Small sample size for speed
    Config.N_FOLDS = 2  # Minimal folds for CV
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing overhead

    # Set up isolated working directory for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_PATH = os.path.join(
        Config.WORKING_DIR, "submission", "submission.csv"
    )

    # Clean up previous run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Initialize logging and seeds
    setup_logging(log_file=os.path.join(Config.WORKING_DIR, "execution.log"))
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 3. Demonstrate Feature Extraction
    # ==========================================
    print("\n--- 1. Testing Feature Extraction ---")

    # Get a sample image path from metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]
    sample_img_path = sample_row["file_path"]

    print(f"Processing image: {sample_img_path}")

    # Test View Generation
    views = get_rotated_views(sample_img_path)
    expected_views = (
        Config.ROTATION_END - Config.ROTATION_START
    ) // Config.ROTATION_STEP

    assert (
        len(views) == expected_views
    ), f"View generation failed. Expected {expected_views}, got {len(views)}"
    print(f"Generated {len(views)} rotated views successfully.")

    # Test Feature Extractor
    print("Initializing FeatureExtractor (loading models)...")
    extractor = FeatureExtractor()
    dino_emb, conv_emb = extractor.extract_features(views)

    # Verify Shapes
    # DINO: (36, 1024), ConvNeXt: (36, 1536)
    assert dino_emb.shape == (
        expected_views,
        1024,
    ), f"DINO shape incorrect: {dino_emb.shape}"
    assert conv_emb.shape == (
        expected_views,
        1536,
    ), f"ConvNeXt shape incorrect: {conv_emb.shape}"
    print("Feature extraction shapes verified.")

    # ==========================================
    # 4. Demonstrate Data Manager
    # ==========================================
    print("\n--- 2. Testing Data Manager ---")
    dm = DataManager()

    # A. Create Densified Training Set
    # This averages 4 orthogonal views to create 9 centroids per image
    print("Generating Densified Training Set...")
    d_dino, d_conv, d_tab, d_ids, d_labels = dm.create_densified_training_set(
        load_cached_data=False
    )

    # Verify Counts
    # Expected: DEBUG_SAMPLE_SIZE * 9 centroids
    expected_densified_count = Config.DEBUG_SAMPLE_SIZE * Config.NUM_TRAIN_CENTROIDS
    assert (
        d_dino.shape[0] == expected_densified_count
    ), f"Densified count mismatch. Expected {expected_densified_count}, got {d_dino.shape[0]}"
    assert d_labels.shape[0] == expected_densified_count
    print(f"Densified Training Set: {d_dino.shape[0]} samples (Correct).")

    # B. Create Canonical Validation Set
    # This averages 4 orthogonal views (starting at 0) to create 1 centroid per image
    print("Generating Canonical Validation Set...")
    c_dino, c_conv, c_tab, c_ids, c_labels = dm.create_canonical_inference_set(
        "val", load_cached_data=False
    )

    # Verify Counts
    # Expected: DEBUG_SAMPLE_SIZE * 1 centroid
    expected_canonical_count = Config.DEBUG_SAMPLE_SIZE
    assert (
        c_dino.shape[0] == expected_canonical_count
    ), f"Canonical count mismatch. Expected {expected_canonical_count}, got {c_dino.shape[0]}"
    print(f"Canonical Validation Set: {c_dino.shape[0]} samples (Correct).")

    # ==========================================
    # 5. Demonstrate Model Factory
    # ==========================================
    print("\n--- 3. Testing Model Factory ---")

    # Build Pipeline
    pipeline = ModelFactory.build_lda_pipeline()
    print("LDA Pipeline built successfully.")

    # Prepare Data for Pipeline (Concatenate features)
    X_train = np.hstack([d_dino, d_conv, d_tab])
    y_train = d_labels
    X_val = np.hstack([c_dino, c_conv, c_tab])

    # Fit
    print("Fitting model on densified data...")
    pipeline.fit(X_train, y_train)

    # Predict
    print("Predicting on canonical data...")
    probs = pipeline.predict_proba(X_val)

    # Verify Output
    assert probs.shape == (
        expected_canonical_count,
        len(pipeline.classes_),
    ), f"Probability shape mismatch: {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1."
    print("Model fit and prediction successful.")

    # ==========================================
    # 6. Demonstrate Workflow (End-to-End)
    # ==========================================
    print("\n--- 4. Testing Full Workflow ---")
    wf = Workflow()

    # Run Cross-Validation
    # This will iterate through folds, training on densified and validating on canonical
    print("Running Cross-Validation...")
    wf.run_cross_validation()

    # Verify Models Saved
    for i in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{i}.joblib")
        assert os.path.exists(model_path), f"Fold {i} model not found at {model_path}"
    print("All fold models saved successfully.")

    # Generate Submission
    print("Generating Submission...")
    wf.generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check rows (should be DEBUG_SAMPLE_SIZE) and columns (id + classes)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"
    assert "id" in df_sub.columns, "Submission missing 'id' column."

    # Check probabilities range
    feature_cols = [c for c in df_sub.columns if c != "id"]
    probs_matrix = df_sub[feature_cols].values
    assert (
        probs_matrix.min() >= 0 and probs_matrix.max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
