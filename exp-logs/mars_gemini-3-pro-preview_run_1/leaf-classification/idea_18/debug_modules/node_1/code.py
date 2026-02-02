import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import shutil

# Import provided library modules
from library import config
from library import data_loader
from library import preprocessing
from library import model


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("--- Starting Library Usage Demonstration ---")
    set_seed(config.SEED)

    # =========================================================================
    # 1. Verify Configuration
    # =========================================================================
    print("\n[1] Verifying Configuration...")
    print(f"    Input Directory: {config.INPUT_DIR}")
    print(f"    Metadata Directory: {config.METADATA_DIR}")
    print(f"    Cache Directory: {config.CACHE_DIR}")

    # Assert critical paths exist (metadata is pre-generated)
    assert os.path.exists(config.TRAIN_DATA_PATH), "Train metadata missing"
    assert os.path.exists(config.VAL_DATA_PATH), "Val metadata missing"
    assert os.path.exists(config.TEST_DATA_PATH), "Test metadata missing"
    print("    Configuration paths verified.")

    # =========================================================================
    # 2. Demonstrate Data Loader
    # =========================================================================
    print("\n[2] Testing Data Loader...")
    # Force loading from CSVs to verify parsing logic (bypass cache for demo)
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.load_datasets(
        load_cached_data=False
    )

    print(f"    Loaded Train shape: {X_train.shape}")
    print(f"    Loaded Val shape:   {X_val.shape}")
    print(f"    Loaded Test shape:  {X_test.shape}")

    # Assertions
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"
    assert X_train.dtypes.iloc[0] == np.float64, "Features should be float64"
    assert len(y_train) == len(X_train), "Mismatch between X_train and y_train"
    print("    Data Loader assertions passed.")

    # =========================================================================
    # 3. Demonstrate Preprocessing
    # =========================================================================
    print("\n[3] Testing Preprocessing Pipeline...")

    # 3a. Unit Test for Float64Pipeline
    print("    Running unit test for Float64Pipeline...")
    dummy_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pipeline = preprocessing.Float64Pipeline()
    pipeline.fit(dummy_data)
    transformed_dummy = pipeline.transform(dummy_data)

    assert transformed_dummy.dtype == np.float64, "Pipeline must return float64"
    # Yeo-Johnson + Standard Scaler on small data usually results in mean ~0
    assert np.allclose(
        transformed_dummy.mean(axis=0), 0, atol=1e-5
    ), "StandardScaler logic check failed"
    print("    Float64Pipeline unit test passed.")

    # 3b. Full Preprocessing
    print("    Running full preprocessing on loaded datasets...")
    # This function handles the pipeline fitting on train and transforming all splits
    # We force re-computation to test the logic
    (
        X_train_trans,
        y_train_trans,
        X_val_trans,
        y_val_trans,
        X_test_trans,
        test_ids_trans,
    ) = preprocessing.get_preprocessed_data(load_cached_data=False)

    assert X_train_trans.dtype == np.float64, "Transformed X_train should be float64"
    assert X_train_trans.shape == X_train.shape, "Shape mismatch after transformation"
    print("    Full preprocessing complete.")

    # =========================================================================
    # 4. Demonstrate Model (DualPrecisionOAS)
    # =========================================================================
    print("\n[4] Testing DualPrecisionOAS Model...")

    clf = model.DualPrecisionOAS()

    # Fit on training data
    print("    Fitting model...")
    clf.fit(X_train_trans, y_train_trans)

    # Check internal parameters
    assert clf.means_.dtype == np.float32, "Model means should be quantized to float32"
    assert (
        clf.precision_.dtype == np.float32
    ), "Model precision matrix should be quantized to float32"
    print("    Model fitted and parameters quantized successfully.")

    # Predict on validation data
    print("    Predicting on validation set...")
    val_probs = clf.predict_proba(X_val_trans)

    # Assertions on predictions
    assert val_probs.shape == (
        len(y_val_trans),
        99,
    ), f"Expected (N_val, 99), got {val_probs.shape}"
    assert np.allclose(
        val_probs.sum(axis=1), 1.0, atol=1e-5
    ), "Probabilities must sum to 1"

    # Calculate Metric
    loss = log_loss(y_val_trans, val_probs, labels=clf.classes_)
    print(f"    Validation Log Loss: {loss:.4f}")

    # Basic sanity check: Random guessing for 99 classes is -ln(1/99) ~= 4.6
    # A trained model should be significantly better
    assert loss < 4.0, "Model performance is worse than random guessing"
    print("    Model performance check passed.")

    # =========================================================================
    # 5. Demonstrate Full Workflow (Train -> Predict -> Submit)
    # =========================================================================
    print("\n[5] Running Full Workflow (train_and_predict)...")

    # This function encapsulates the final retraining on Train+Val and generating submission
    model.train_and_predict()

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Format
    df_sub = pd.read_csv(submission_path)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    expected_cols = 1 + 99  # id + 99 classes
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {df_sub.shape[1]}"
    assert config.ID_COL in df_sub.columns, f"Missing '{config.ID_COL}' column"
    assert (
        df_sub[config.ID_COL].dtype == np.int64
        or df_sub[config.ID_COL].dtype == np.int32
    ), "ID column should be integer"

    # Check if probabilities are valid
    prob_cols = [c for c in df_sub.columns if c != config.ID_COL]
    probs = df_sub[prob_cols].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("    Submission format verified.")
    print("\n--- Demonstration Complete: All checks passed ---")


if __name__ == "__main__":
    run_demo()
