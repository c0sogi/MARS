import os
import sys
import numpy as np
import pandas as pd
import ase.io
import torch
import shutil

# Import library components
from library.config import Config
from library.data_manager import load_metadata
from library.embeddings import MatGLExtractor
from library.feature_pipeline import prepare_features
from library.trainer import train_xgboost_models, predict
from library.utils import save_submission


def run_demo():
    print("=" * 60)
    print("STARTING LIBRARY DEMONSTRATION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Use a separate working directory for this demo to avoid messing up real cache
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config class attributes
    Config.WORKING_DIR = demo_working_dir
    Config.TRAIN_FEATURES_PATH = os.path.join(
        demo_working_dir, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(demo_working_dir, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_working_dir, "test_features.parquet")

    # Set extremely lightweight parameters for XGBoost
    Config.XGB_PARAMS = {
        "n_estimators": 2,
        "learning_rate": 0.1,
        "max_depth": 2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "n_jobs": 1,
        "random_state": 42,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }

    # We will manually control sample_size in function calls,
    # but setting this ensures safety if defaults are used.
    Config.SAMPLE_SIZE = 10

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"XGBoost params set to: {Config.XGB_PARAMS}")

    # -------------------------------------------------------------------------
    # 2. Data Manager Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Manager (load_metadata)...")

    sample_n = 10
    df_train_sample = load_metadata(split="train", sample_size=sample_n)

    print(f"Loaded training metadata sample shape: {df_train_sample.shape}")

    # Validation
    assert (
        len(df_train_sample) == sample_n
    ), f"Expected {sample_n} samples, got {len(df_train_sample)}"
    assert "file_path" in df_train_sample.columns, "Metadata missing 'file_path' column"
    assert "id" in df_train_sample.columns, "Metadata missing 'id' column"
    print("Data Manager validation passed.")

    # -------------------------------------------------------------------------
    # 3. GNN Embedding Extraction Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Testing MatGLExtractor on a single structure...")

    # Get the first file path from the sample
    rel_path = df_train_sample.iloc[0]["file_path"]
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    if os.path.exists(full_path):
        print(f"Reading structure from: {full_path}")
        atoms = ase.io.read(full_path, format="aims")

        # Instantiate extractor
        extractor = MatGLExtractor()

        # Process structure
        features = extractor.process_structure(atoms)

        print(f"Extracted feature vector shape: {features.shape}")

        # Validation: M3GNet usually outputs 64-dim embeddings.
        # We pool mean, std, range -> 64 * 3 = 192 dimensions.
        expected_dim = 192
        assert features.shape == (
            expected_dim,
        ), f"Expected feature dimension {expected_dim}, got {features.shape[0]}"
        assert not np.all(
            features == 0
        ), "Feature vector is all zeros (extraction likely failed)"
        print("MatGLExtractor validation passed.")
    else:
        raise FileNotFoundError(f"Sample geometry file not found: {full_path}")

    # -------------------------------------------------------------------------
    # 4. Feature Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing Feature Pipeline (prepare_features)...")

    # We force load_cached_data=False to ensure the pipeline actually runs extraction
    X_train, y_train, ids_train = prepare_features(
        split="train", sample_size=sample_n, load_cached_data=False
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Validation
    assert len(X_train) == sample_n
    assert len(y_train) == sample_n
    assert len(ids_train) == sample_n
    # Check if GNN features are present (columns starting with 'gnn_')
    gnn_cols = [c for c in X_train.columns if c.startswith("gnn_")]
    assert len(gnn_cols) > 0, "No GNN features found in X_train"
    # Check if physical features are present (e.g., 'volume')
    assert "volume" in X_train.columns, "Physical feature 'volume' missing"

    print("Feature Pipeline validation passed.")

    # -------------------------------------------------------------------------
    # 5. Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Testing Model Training (train_xgboost_models)...")

    # Note: train_xgboost_models internally calls prepare_features for train and val.
    # We use a small sample size for both.
    models, feature_cols, val_rmsle = train_xgboost_models(
        sample_size=sample_n,
        load_cached_data=True,  # Can use cache now since we just generated it (though filenames might differ if sample size logic in cache naming is strict)
        # Actually, let's set False to be safe and robust for this demo script.
    )

    print(f"Trained models for targets: {list(models.keys())}")
    print(f"Validation RMSLE: {val_rmsle}")

    # Validation
    for target in Config.TARGET_COLS:
        assert target in models, f"Model for {target} not found"
        assert models[target] is not None, f"Model for {target} is None"
    assert isinstance(val_rmsle, float), "Validation RMSLE is not a float"

    print("Training validation passed.")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Testing Inference (predict)...")

    test_sample_n = 5
    ids_test, preds_test = predict(
        models=models,
        feature_columns=feature_cols,
        sample_size=test_sample_n,
        load_cached_data=False,
    )

    print(f"Test IDs shape: {ids_test.shape}")
    print(f"Predictions shape: {preds_test.shape}")
    print(f"Sample predictions:\n{preds_test[:2]}")

    # Validation
    assert len(ids_test) == test_sample_n
    assert preds_test.shape == (test_sample_n, 2)
    assert not np.any(np.isnan(preds_test)), "Predictions contain NaNs"
    # Physical check: Energies should be non-negative (handled by pipeline, but good to check)
    assert np.all(preds_test >= 0), "Predictions contain negative values"

    print("Inference validation passed.")

    # -------------------------------------------------------------------------
    # 7. Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[7] Testing Submission Generation (save_submission)...")

    submission_filename = "demo_submission.csv"
    save_submission(ids_test, preds_test, filename=submission_filename)

    submission_path = os.path.join(Config.SUBMISSION_DIR, submission_filename)
    assert os.path.exists(
        submission_path
    ), f"Submission file not created at {submission_path}"

    # Read back to verify format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission file content head:\n{df_sub.head()}")

    expected_cols = ["id"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print("Submission validation passed.")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
