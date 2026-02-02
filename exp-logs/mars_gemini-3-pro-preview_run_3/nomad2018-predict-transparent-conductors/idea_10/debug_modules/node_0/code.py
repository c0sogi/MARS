import os
import shutil
import numpy as np
import pandas as pd
from library.data_factory import CrystalDataHandler
from library.model_factory import XGBoostRegressorWrapper
from library.utils import save_submission

# Constants
CACHE_DIR = "./working/demo_execution"
SUBMISSION_FILE = "./working/demo_submission.csv"
RANDOM_SEED = 42
MAX_SAMPLES = 50  # Reduced for speed in this demonstration
N_ESTIMATORS = 100  # Reduced for speed


def main():
    print("Starting demo execution...")

    # Clean up previous run if exists
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    # ---------------------------------------------------------
    # 1. Data Loading and Feature Generation
    # ---------------------------------------------------------
    print("\n--- 1. Loading Data and Generating Features ---")
    data_handler = CrystalDataHandler(metadata_dir="./metadata", cache_dir=CACHE_DIR)

    # Load a subset of data. This triggers feature extraction (Physical + RDF)
    # and caching in CACHE_DIR.
    (X_train, y_train), (X_val, y_val), X_test, test_ids = data_handler.load_data(
        load_cached_data=False, max_samples=MAX_SAMPLES  # Force generation for demo
    )

    # Verify data shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    assert X_train.shape[0] == MAX_SAMPLES, "X_train has incorrect number of samples"
    assert y_train.shape[0] == MAX_SAMPLES, "y_train has incorrect number of samples"
    assert X_val.shape[0] == MAX_SAMPLES, "X_val has incorrect number of samples"
    assert X_test.shape[0] == MAX_SAMPLES, "X_test has incorrect number of samples"
    assert X_train.shape[1] > 0, "Features were not generated correctly"

    # ---------------------------------------------------------
    # 2. Model Training and Prediction
    # ---------------------------------------------------------
    print("\n--- 2. Training Models and Predicting ---")

    targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    predictions = {}

    for target in targets:
        print(f"\nProcessing target: {target}")

        # Instantiate the wrapper with reduced estimators for speed
        model_wrapper = XGBoostRegressorWrapper(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_SEED,
        )

        # Select specific target column (already log-transformed by DataHandler)
        y_tr_target = y_train[target]
        y_val_target = y_val[target]

        # Fit the model
        # The wrapper handles feature pruning (removing constant columns) internally
        model_wrapper.fit_model(X_train, y_tr_target, X_val, y_val_target)

        # Verify valid features were selected
        assert (
            len(model_wrapper.valid_features) > 0
        ), "No valid features found after pruning"
        print(f"Selected {len(model_wrapper.valid_features)} features for {target}")

        # Predict on test set (returns log-space predictions)
        log_preds = model_wrapper.predict_model(X_test)

        # Verify prediction shape
        assert log_preds.shape[0] == len(X_test), "Prediction shape mismatch"

        # Inverse transform (expm1) to get original scale
        # Note: DataHandler applied log1p, so we apply expm1
        predictions[target] = np.expm1(log_preds)

        # Basic sanity check on predictions (should be non-negative)
        assert np.all(
            predictions[target] >= 0
        ), f"Negative predictions found for {target}"

    # ---------------------------------------------------------
    # 3. Submission Generation
    # ---------------------------------------------------------
    print("\n--- 3. Saving Submission ---")

    save_submission(
        ids=test_ids,
        formation_energy=predictions["formation_energy_ev_natom"],
        bandgap_energy=predictions["bandgap_energy_ev"],
        filename=SUBMISSION_FILE,
    )

    # Verify submission file exists
    assert os.path.exists(SUBMISSION_FILE), "Submission file was not created"

    # Verify submission content format
    df_sub = pd.read_csv(SUBMISSION_FILE)
    print(f"Submission head:\n{df_sub.head()}")

    assert df_sub.shape == (MAX_SAMPLES, 3), "Submission has incorrect shape"
    assert list(df_sub.columns) == [
        "id",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
    ], "Incorrect columns in submission"

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
