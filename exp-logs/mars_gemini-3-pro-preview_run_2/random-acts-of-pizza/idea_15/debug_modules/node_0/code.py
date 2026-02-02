import os
import shutil
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Import library modules
from library.config import Config
from library.data_loader import DataLoader
from library.text_encoder import TextEncoder
from library.tabular_processor import TabularProcessor
import library.model_factory as model_factory
from library.trainer import Trainer
from library.inference import Predictor


def main():
    print("Initializing Demo Script...")

    # ==========================================
    # 1. Configuration Override for Speed & Demo
    # ==========================================
    print("Patching Config for fast demonstration...")

    # Define demo-specific directories to avoid overwriting production artifacts
    DEMO_WORKING_DIR = "./working/demo_execution"
    DEMO_OUTPUT_DIR = "./working/demo_submission"

    # Clean up previous demo runs if any
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    if os.path.exists(DEMO_OUTPUT_DIR):
        shutil.rmtree(DEMO_OUTPUT_DIR)
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    # Patch Config paths
    # Note: We must update derived paths manually as they were initialized at import time
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.OUTPUT_DIR = DEMO_OUTPUT_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_OUTPUT_DIR, "demo_submission.csv")

    Config.TRAIN_FEATURES_PATH = os.path.join(
        DEMO_WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(DEMO_WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_WORKING_DIR, "test_features.parquet")

    Config.TRAIN_EMBEDDINGS_PATH = os.path.join(
        DEMO_WORKING_DIR, "train_embeddings.npy"
    )
    Config.VAL_EMBEDDINGS_PATH = os.path.join(DEMO_WORKING_DIR, "val_embeddings.npy")
    Config.TEST_EMBEDDINGS_PATH = os.path.join(DEMO_WORKING_DIR, "test_embeddings.npy")

    # Reduce computational load for the demo
    Config.N_SPLITS = 2  # Use 2 folds instead of 5
    Config.PLS_N_COMPONENTS_GRID = [2]  # Single value for grid search
    Config.LR_C_GRID = [1.0]  # Single value for grid search
    Config.LR_CLASS_WEIGHT_GRID = [None]  # Single value
    Config.N_BAGGING_ESTIMATORS = 2  # Minimal ensemble size

    print("Config patched successfully.")

    # ==========================================
    # 2. Data Loading & Subsetting
    # ==========================================
    print("\n--- Testing DataLoader ---")
    # Load full data first (ignoring cache to ensure we read raw files)
    df_train_full, df_val_full, df_test_full = DataLoader.load_data(
        load_cached_data=False
    )

    print(f"Original Train shape: {df_train_full.shape}")

    # Validation
    assert not df_train_full.empty, "Training dataframe is empty"
    assert "requester_received_pizza" in df_train_full.columns, "Target column missing"

    # Create a small subset for rapid execution
    SUBSET_SIZE = 50
    print(f"Subsetting data to {SUBSET_SIZE} samples for speed...")

    df_train = df_train_full.head(SUBSET_SIZE).copy()
    df_val = df_val_full.head(SUBSET_SIZE).copy()
    df_test = df_test_full.head(SUBSET_SIZE).copy()

    # Save subsets to the patched cache locations.
    # The Trainer will load these files when we pass load_cached_data=True.
    df_train.to_parquet(Config.TRAIN_FEATURES_PATH, index=False)
    df_val.to_parquet(Config.VAL_FEATURES_PATH, index=False)
    df_test.to_parquet(Config.TEST_FEATURES_PATH, index=False)

    print("Subset data saved to cache.")

    # ==========================================
    # 3. Component Testing: TextEncoder
    # ==========================================
    print("\n--- Testing TextEncoder ---")
    encoder = TextEncoder()

    # Use a temp path for this specific test to not interfere with Trainer's cache
    temp_emb_path = os.path.join(DEMO_WORKING_DIR, "temp_text_emb.npy")

    # Generate embeddings for the subset
    embeddings = encoder.encode(df_train, temp_emb_path, load_cached_data=False)

    print(f"Generated embeddings shape: {embeddings.shape}")

    # Verify shape (N_samples, 768 for MPNet)
    assert embeddings.shape == (len(df_train), 768), "Incorrect embedding shape"
    # Verify normalization (L2 norm should be approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not normalized"

    # ==========================================
    # 4. Component Testing: TabularProcessor
    # ==========================================
    print("\n--- Testing TabularProcessor ---")
    processor = TabularProcessor()

    X_tab = processor.process(df_train)
    print(f"Tabular features shape: {X_tab.shape}")

    # Verify shape (N_samples, N_numeric_cols)
    expected_cols = len(Config.NUMERIC_COLS)
    assert X_tab.shape == (len(df_train), expected_cols), "Incorrect tabular shape"
    assert not np.isnan(X_tab).any(), "Tabular features contain NaNs"

    # ==========================================
    # 5. Component Testing: Model Factory
    # ==========================================
    print("\n--- Testing Model Factory ---")

    pls = model_factory.get_pls_transformer(n_components=2)
    assert isinstance(pls, BaseEstimator), "PLS object is not a valid estimator"

    scaler = model_factory.get_scaler()
    assert hasattr(scaler, "fit_transform"), "Scaler missing fit_transform"

    clf = model_factory.get_classifier(n_estimators=2)
    assert hasattr(clf, "fit"), "Classifier missing fit method"

    print("Model factory components verified.")

    # ==========================================
    # 6. Pipeline Execution: Trainer
    # ==========================================
    print("\n--- Testing Trainer (Training Phase) ---")
    trainer = Trainer()

    # Execute full training loop on the subset
    # This will:
    # 1. Load the subset parquet files we saved earlier.
    # 2. Compute/Cache embeddings for them.
    # 3. Run 2-Fold CV with the simplified grid.
    # 4. Save model artifacts to ./working/demo_execution/models
    # 5. Generate a submission file.
    trainer.train_and_submit(load_cached_data=True)

    # Verify Artifacts
    models_dir = os.path.join(DEMO_WORKING_DIR, "models")
    expected_artifacts = [
        "pls_fold_0.joblib",
        "clf_fold_0.joblib",
        "tab_scaler_fold_0.joblib",
        "pls_fold_1.joblib",
        "clf_fold_1.joblib",
        "tab_scaler_fold_1.joblib",
    ]

    for artifact in expected_artifacts:
        path = os.path.join(models_dir, artifact)
        assert os.path.exists(path), f"Model artifact {artifact} missing."

    print("Training completed and artifacts verified.")

    # ==========================================
    # 7. Pipeline Execution: Inference
    # ==========================================
    print("\n--- Testing Predictor (Inference Phase) ---")

    # Instantiate Predictor to verify independent inference logic
    predictor = Predictor()

    # Ensure it looks for models in the demo directory
    # (Trainer and Predictor both use Config.WORKING_DIR/models, which we patched)
    assert predictor.models_dir == models_dir

    # Run inference
    predictor.predict_submission(load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Check format
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect columns"
    assert len(df_sub) == len(
        df_test
    ), f"Submission row count mismatch. Expected {len(df_test)}, got {len(df_sub)}"

    # Check probability range
    preds = df_sub["requester_received_pizza"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\nAll demonstrations and validations passed successfully.")


if __name__ == "__main__":
    main()
