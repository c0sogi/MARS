import os
import shutil
import numpy as np
import pandas as pd
import joblib

# Import library modules
# We import Config first to patch it before other modules use it
from library.config import Config
import library.utils
import library.data_loader
import library.feature_engineering
import library.preprocessing
import library.model_factory
import library.trainer
import library.inference


def main():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Configuration & Patching for Speed
    # ==========================================
    print("Patching Configuration for fast demonstration...")

    # Define a separate working directory for the demo to avoid conflicts with real runs
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR)

    # Patch Config paths to point to the demo directory
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_CHECKPOINT_DIR = os.path.join(DEMO_WORKING_DIR, "models")

    # Update Cache Paths in Config to point to new working dir
    Config.CACHE_TRAIN_ANCHOR_TITLE = os.path.join(
        DEMO_WORKING_DIR, "train_anchor_title.npy"
    )
    Config.CACHE_TRAIN_ANCHOR_BODY = os.path.join(
        DEMO_WORKING_DIR, "train_anchor_body.npy"
    )
    Config.CACHE_TRAIN_AUX_GLOBAL = os.path.join(
        DEMO_WORKING_DIR, "train_aux_global.npy"
    )
    Config.CACHE_TRAIN_AUX_HOOK = os.path.join(DEMO_WORKING_DIR, "train_aux_hook.npy")
    Config.CACHE_TEST_ANCHOR_TITLE = os.path.join(
        DEMO_WORKING_DIR, "test_anchor_title.npy"
    )
    Config.CACHE_TEST_ANCHOR_BODY = os.path.join(
        DEMO_WORKING_DIR, "test_anchor_body.npy"
    )
    Config.CACHE_TEST_AUX_GLOBAL = os.path.join(DEMO_WORKING_DIR, "test_aux_global.npy")
    Config.CACHE_TEST_AUX_HOOK = os.path.join(DEMO_WORKING_DIR, "test_aux_hook.npy")

    # Ensure directories exist
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_CHECKPOINT_DIR, exist_ok=True)

    # Reduce computational load for the demo
    Config.N_FOLDS = 2  # Use only 2 folds instead of 5
    Config.N_ESTIMATORS = 2  # Use only 2 estimators in BaggingClassifier
    Config.PARAM_GRID = {"C": [1.0]}  # Disable GridSearch by providing a single param

    # Patch load_data to subsample datasets
    # This ensures we don't compute embeddings for thousands of texts during the demo
    original_load_data = library.data_loader.load_data

    def mocked_load_data(load_cached_data=True):
        # We ignore load_cached_data=True to force loading raw data and subsampling it
        print("Mocked load_data: Loading raw data and subsampling...")
        train_df, val_df, test_df = original_load_data(load_cached_data=False)

        # Subsample: 50 train, 20 val, 20 test
        train_sub = train_df.head(50).copy()
        val_sub = val_df.head(20).copy()
        test_sub = test_df.head(20).copy()

        print(
            f"Subsampled shapes - Train: {train_sub.shape}, Val: {val_sub.shape}, Test: {test_sub.shape}"
        )
        return train_sub, val_sub, test_sub

    # Apply the patch to both the data_loader module and the feature_engineering module
    # (since feature_engineering imports load_data)
    library.data_loader.load_data = mocked_load_data
    library.feature_engineering.load_data = mocked_load_data

    # Set Seed for reproducibility
    library.utils.set_seed(Config.SEED)

    # ==========================================
    # 2. Demonstrate Feature Engineering
    # ==========================================
    print("\n=== Step 2: Feature Engineering ===")
    fe = library.feature_engineering.FeatureEngineer()

    # Build features (this will trigger embedding generation on the subset)
    # We set load_cached_data=False to force generation in our demo dir
    feature_set = fe.build_feature_set(load_cached_data=False)

    # Verification
    print("Verifying Feature Set structure...")
    assert "train" in feature_set
    assert "val" in feature_set
    assert "test" in feature_set

    # Check dimensions for Train
    train_feats = feature_set["train"]
    n_train = 50  # Based on our subsampling

    # Verify lengths matches subsample size
    assert len(train_feats["y"]) == n_train

    # Verify Embedding Dimensions
    # MiniLM (Anchor) -> 384 dim
    # MPNet (Aux) -> 768 dim
    assert train_feats["anchor_title"].shape == (n_train, 384)
    assert train_feats["anchor_body"].shape == (n_train, 384)
    assert train_feats["aux_global"].shape == (n_train, 768)
    assert train_feats["aux_hook"].shape == (n_train, 768)

    # Verify Metadata Dimensions (10 numeric columns defined in Config)
    assert train_feats["metadata"].shape == (n_train, 10)

    print("Feature Engineering verified successfully.")

    # ==========================================
    # 3. Demonstrate Preprocessing
    # ==========================================
    print("\n=== Step 3: Preprocessing ===")
    preprocessor = library.preprocessing.HAMFPreprocessor()

    # Fit on training data
    preprocessor.fit(train_feats)
    assert preprocessor.is_fitted is True

    # Transform training data
    X_train_processed = preprocessor.transform(train_feats)

    # Verify Combined Dimensions
    # Calculation:
    #   384 (Title Anchor)
    # + 384 (Body Anchor)
    # + 50 (Global PCA, reduced from 768)
    # + 20 (Hook PCA, reduced from 768)
    # + 10 (Metadata)
    # = 848
    expected_dim = 384 + 384 + 50 + 20 + 10
    print(f"Processed Feature Shape: {X_train_processed.shape}")
    assert X_train_processed.shape == (n_train, expected_dim)

    print("Preprocessing verified successfully.")

    # ==========================================
    # 4. Demonstrate Training Pipeline
    # ==========================================
    print("\n=== Step 4: Training Pipeline ===")
    trainer = library.trainer.Trainer()

    # Run training (uses patched load_data and config)
    # This handles CV, model fitting, and saving artifacts
    trainer.run_training(load_cached_data=False)

    # Verify Artifacts
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(
            Config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold}.joblib"
        )
        proc_path = os.path.join(
            Config.MODEL_CHECKPOINT_DIR, f"processor_fold_{fold}.joblib"
        )
        assert os.path.exists(model_path), f"Model for fold {fold} missing"
        assert os.path.exists(proc_path), f"Preprocessor for fold {fold} missing"

    # Verify Submission from Trainer
    assert os.path.exists(Config.SUBMISSION_PATH)
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == 20  # We subsampled test to 20
    assert "request_id" in sub_df.columns
    assert "requester_received_pizza" in sub_df.columns

    print("Training pipeline executed and verified.")

    # ==========================================
    # 5. Demonstrate Inference Manager
    # ==========================================
    print("\n=== Step 5: Inference Manager ===")
    # Delete previous submission to verify InferenceManager creates a new one
    os.remove(Config.SUBMISSION_PATH)

    inference_mgr = library.inference.InferenceManager()

    # Run prediction using the models trained in Step 4
    # We use load_cached_data=True here because the embeddings were generated and cached in Step 2/4
    inference_mgr.predict_test_set(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH)
    sub_df_inf = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df_inf) == 20

    # Check values are valid probabilities
    preds = sub_df_inf["requester_received_pizza"]
    assert preds.min() >= 0.0 and preds.max() <= 1.0

    print("Inference Manager executed and verified.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
