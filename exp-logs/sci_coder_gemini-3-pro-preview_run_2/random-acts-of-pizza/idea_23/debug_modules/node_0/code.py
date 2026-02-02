import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import torch

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import DataLoader
from library.feature_generator import FeatureGenerator
from library.custom_ensemble import StratifiedRandomSubspaceEnsemble
from library.trainer import Trainer
from library.inference import InferenceRunner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demo
    set_seed(42)

    print("Configuring environment for fast execution...")
    Config.MAX_SAMPLES = 50  # Use only 50 samples per split
    Config.N_FOLDS = 2  # Use only 2 folds
    Config.N_ESTIMATORS = 5  # Only 5 base learners per ensemble
    Config.LR_C_GRID = [1.0]  # No grid search for C
    Config.LR_CLASS_WEIGHTS = [None]  # No grid search for class weights
    Config.WORKING_DIR = "./working/demo_execution"  # Separate working dir
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Demonstrate Data Loading
    print("\n--- Testing DataLoader ---")
    data_loader = DataLoader()
    # Force reload from source to respect MAX_SAMPLES
    df_train, df_val, df_test = data_loader.load_data(load_cached_data=False)

    # Verification
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    assert (
        len(df_train) == Config.MAX_SAMPLES
    ), f"Expected {Config.MAX_SAMPLES} train samples, got {len(df_train)}"
    assert (
        Config.TARGET_COL in df_train.columns
    ), "Target column missing in training data"
    assert Config.ID_COL in df_test.columns, "ID column missing in test data"
    print("DataLoader verification successful.")

    # 3. Demonstrate Feature Generation
    print("\n--- Testing FeatureGenerator ---")
    feature_gen = FeatureGenerator()

    # Generate Embeddings (this uses the SentenceTransformer)
    # Note: This might take a few seconds to load the model
    print("Generating embeddings for training subset...")
    X_text_train = feature_gen.generate_embeddings(
        df_train, "train", load_cached_data=False
    )

    # Extract Tabular Features
    print("Extracting tabular features...")
    X_tab_train = feature_gen.extract_tabular_features(df_train)

    # Verification
    print(f"Text Embeddings shape: {X_text_train.shape}")
    print(f"Tabular Features shape: {X_tab_train.shape}")

    assert X_text_train.shape[0] == len(df_train), "Embedding count mismatch"
    assert (
        X_text_train.shape[1] == Config.EMBEDDING_DIM
    ), f"Expected embedding dim {Config.EMBEDDING_DIM}, got {X_text_train.shape[1]}"
    assert X_tab_train.shape[0] == len(df_train), "Tabular feature count mismatch"
    assert X_tab_train.shape[1] == len(
        Config.NUMERIC_COLS
    ), "Tabular feature dimension mismatch"
    print("FeatureGenerator verification successful.")

    # 4. Demonstrate Custom Ensemble Logic
    print("\n--- Testing StratifiedRandomSubspaceEnsemble ---")
    # Create synthetic data for pure logic test
    n_synth = 20
    n_text_feat = 100
    n_tab_feat = 5

    X_text_synth = np.random.rand(n_synth, n_text_feat).astype(np.float32)
    X_tab_synth = np.random.rand(n_synth, n_tab_feat).astype(np.float32)
    y_synth = np.random.randint(0, 2, size=n_synth)

    ensemble = StratifiedRandomSubspaceEnsemble(
        n_estimators=5, subspace_fraction=0.5, random_state=42, verbose=0
    )

    # Fit
    ensemble.fit(X_text_synth, X_tab_synth, y_synth)

    # Predict
    probs = ensemble.predict_proba(X_text_synth, X_tab_synth)
    preds = ensemble.predict(X_text_synth, X_tab_synth)

    # Verification
    assert (
        len(ensemble.estimators_) == 5
    ), "Ensemble did not train correct number of estimators"
    assert len(ensemble.feature_masks_) == 5, "Ensemble did not store feature masks"
    # Check subspace size: 50% of 100 = 50
    assert len(ensemble.feature_masks_[0]) == 50, "Subspace sampling size incorrect"

    assert probs.shape == (n_synth, 2), "Probability output shape incorrect"
    assert preds.shape == (n_synth,), "Prediction output shape incorrect"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"
    print("Ensemble logic verification successful.")

    # 5. Demonstrate Full Training Pipeline
    print("\n--- Running Trainer (Cross-Validation) ---")
    trainer = Trainer()
    # This will run the full CV loop using the reduced parameters in Config
    trainer.run_cross_validation()

    # Verify Artifacts
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    expected_models = [f"model_fold_{i}.joblib" for i in range(Config.N_FOLDS)]
    expected_scalers = [f"scaler_fold_{i}.joblib" for i in range(Config.N_FOLDS)]

    for m in expected_models:
        assert os.path.exists(
            os.path.join(models_dir, m)
        ), f"Model artifact {m} missing"
    for s in expected_scalers:
        assert os.path.exists(
            os.path.join(models_dir, s)
        ), f"Scaler artifact {s} missing"

    print("Training pipeline completed and artifacts verified.")

    # 6. Demonstrate Inference Pipeline
    print("\n--- Running InferenceRunner ---")
    inference_runner = InferenceRunner()
    inference_runner.generate_submission()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns incorrect"
    assert (
        len(df_sub) == Config.MAX_SAMPLES
    ), f"Submission length mismatch. Expected {Config.MAX_SAMPLES}, got {len(df_sub)}"
    assert df_sub[Config.TARGET_COL].dtype == float, "Prediction column should be float"

    print("\nInference pipeline completed and submission verified.")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
