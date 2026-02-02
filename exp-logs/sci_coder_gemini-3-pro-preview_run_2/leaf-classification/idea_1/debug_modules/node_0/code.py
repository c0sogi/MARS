import os
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.data_loader import LeafDataLoader
from library.preprocessing import FeatureScaler
from library.model import LogisticBaseline, generate_submission
from library.evaluation import evaluate_model


def run_demo():
    # ==========================================
    # 0. Setup
    # ==========================================
    print(">>> Starting Leaf Classification Demo")
    np.random.seed(42)

    # Clean up working directory for a fresh demo run (optional, ensures we test processing logic)
    # We remove cached parquet/npy files to demonstrate the 'load_cached_data=False' path initially
    # or just rely on the overwrite logic. For this demo, we'll force non-cached loading
    # to verify the raw data processing capability.
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    print("\n[1] Loading Data...")
    loader = LeafDataLoader()

    # Load data without cache to demonstrate processing from metadata
    data = loader.load_data(load_cached_data=False)

    X_train, y_train, train_ids = data["train"]
    X_val, y_val, val_ids = data["val"]
    X_test, test_ids = data["test"]
    encoder = data["encoder"]

    # Logic Verification: Check Data Shapes
    # Based on metadata analysis: 192 features (64 margin + 64 shape + 64 texture)
    n_features = 192
    print(
        f"    Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    assert (
        X_train.shape[1] == n_features
    ), f"Expected {n_features} features, got {X_train.shape[1]}"
    assert X_val.shape[1] == n_features, "Validation feature count mismatch"
    assert X_test.shape[1] == n_features, "Test feature count mismatch"
    assert len(X_train) == len(y_train), "Mismatch between Train features and labels"
    assert len(X_val) == len(y_val), "Mismatch between Val features and labels"

    print("    Data loading and shape verification successful.")

    # ==========================================
    # 2. Preprocessing (Scaling)
    # ==========================================
    print("\n[2] Scaling Features...")
    scaler = FeatureScaler()

    # Scale features
    X_train_s, X_val_s, X_test_s = scaler.scale_features(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Logic Verification: Check Statistics
    # Standard Scaler should result in mean ~0 and std ~1 for training data
    train_mean = np.mean(X_train_s)
    train_std = np.std(X_train_s)

    print(f"    Scaled Train Mean: {train_mean:.4f} (Expected ~0)")
    print(f"    Scaled Train Std:  {train_std:.4f} (Expected ~1)")

    assert np.abs(train_mean) < 1e-2, "Feature scaling failed: Mean is not approx 0"
    assert np.abs(train_std - 1.0) < 1e-2, "Feature scaling failed: Std is not approx 1"

    print("    Preprocessing verification successful.")

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("\n[3] Training Model...")

    # Optimize for speed: Reduce max_iter for demonstration
    fast_params = Config.MODEL_PARAMS.copy()
    fast_params["max_iter"] = 50  # Reduced from 2000 for quick execution
    fast_params["verbose"] = 0

    model = LogisticBaseline(params=fast_params)
    model.train(X_train_s, y_train, X_val_s, y_val)

    # Logic Verification: Check if model learned classes
    n_classes = len(encoder.classes_)
    assert (
        len(model.classes_) == n_classes
    ), f"Model did not learn all classes. Expected {n_classes}, got {len(model.classes_)}"

    print("    Model training successful.")

    # ==========================================
    # 4. Evaluation
    # ==========================================
    print("\n[4] Evaluating Model...")

    # Evaluate on validation set
    loss = evaluate_model(model, X_val_s, y_val)

    # Logic Verification: Loss validity
    assert isinstance(loss, float), "Loss should be a float"
    assert loss > 0, "Log loss should be positive"

    print(f"    Validation Log Loss verified: {loss:.4f}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\n[5] Generating Submission...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    generate_submission(model, X_test_s, test_ids, encoder, submission_path)

    # Logic Verification: Check output file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission shape: {df_sub.shape}")

    # Expected columns: 'id' + 99 species
    expected_cols = 1 + n_classes
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Submission has incorrect columns. Expected {expected_cols}, got {df_sub.shape[1]}"
    assert df_sub.shape[0] == len(
        X_test
    ), f"Submission row count mismatch. Expected {len(X_test)}, got {df_sub.shape[0]}"

    # Check probability constraints (0 <= p <= 1)
    # Exclude 'id' column
    probs = df_sub.iloc[:, 1:].values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]"

    print("    Submission verification successful.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
