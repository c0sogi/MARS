import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_log_loss, save_submission
from library.data_loader import load_dataset
from library.preprocessor import preprocess_data
from library.fisher_gp_model import FisherBayesianEnsemble


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("--- 1. Configuration & Setup ---")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Optimize Config for speed during this demo
    # We reduce the GPC complexity to ensure the script finishes quickly.
    # The full dataset is small enough (approx 700 samples) to load quickly.
    print("Overriding Config parameters for speed...")
    Config.GPC_N_RESTARTS_OPTIMIZER = 0  # No restarts for the optimizer
    Config.GPC_MAX_ITER_PREDICT = 10  # Limit prediction iterations
    Config.DEBUG = False  # Use full data to ensure all classes are present

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- 2. Data Loading ---")

    # Load dataset (force reload to demonstrate the full process)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=False
    )

    # Verification
    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Classes: {len(classes)}")

    assert (
        X_train.shape[1] == Config.N_FEATURES
    ), "Incorrect feature count in training data"
    assert (
        len(y_train) == X_train.shape[0]
    ), "Mismatch between training samples and labels"
    assert (
        len(classes) == Config.N_CLASSES
    ), f"Expected {Config.N_CLASSES} classes, found {len(classes)}"

    # -------------------------------------------------------------------------
    # 3. Preprocessing
    # -------------------------------------------------------------------------
    print("\n--- 3. Preprocessing ---")

    # Apply PowerTransformer and StandardScaler
    # We force reprocessing to test the pipeline logic
    X_train_proc, X_val_proc, X_test_proc = preprocess_data(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Verification: Check if standardization worked (mean approx 0, std approx 1)
    # We check a random feature column
    feat_idx = 0
    mean_val = np.mean(X_train_proc[:, feat_idx])
    std_val = np.std(X_train_proc[:, feat_idx])

    print(
        f"Feature {feat_idx} stats after preprocessing: Mean={mean_val:.4f}, Std={std_val:.4f}"
    )
    assert np.abs(mean_val) < 1e-6, "Preprocessing failed: Mean is not approx 0"
    assert np.abs(std_val - 1.0) < 1e-6, "Preprocessing failed: Std is not approx 1"

    # -------------------------------------------------------------------------
    # 4. Model Training (Fisher-Bayesian Ensemble)
    # -------------------------------------------------------------------------
    print("\n--- 4. Model Training ---")

    # Instantiate the model explicitly to demonstrate class usage
    # We use the optimized parameters set in Config
    model = FisherBayesianEnsemble(
        lda_n_components=Config.LDA_N_COMPONENTS,
        gpc_n_restarts=Config.GPC_N_RESTARTS_OPTIMIZER,
        gpc_max_iter_predict=Config.GPC_MAX_ITER_PREDICT,
        random_state=Config.SEED,
    )

    # Fit the model
    # This involves LDA projection followed by GPC fitting
    model.fit(X_train_proc, y_train)

    # -------------------------------------------------------------------------
    # 5. Evaluation
    # -------------------------------------------------------------------------
    print("\n--- 5. Evaluation ---")

    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val_proc)

    # Verify probabilities
    assert val_probs.shape == (
        len(y_val),
        len(classes),
    ), "Probability output shape mismatch"
    assert np.all(val_probs >= 0) and np.all(
        val_probs <= 1
    ), "Probabilities out of [0, 1] range"

    # Calculate Log Loss using the provided utility
    # We pass the range of class indices as labels
    labels_indices = np.arange(len(classes))
    loss = calculate_log_loss(y_val, val_probs, labels=labels_indices)

    print(f"Validation Log Loss: {loss:.5f}")
    assert isinstance(loss, float), "Log loss is not a float"
    assert loss > 0, "Log loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n--- 6. Metric Logic Verification ---")

    # Verify calculate_log_loss logic with a controlled dummy example
    # Case: Perfect prediction should yield near-zero loss (clipped by epsilon)
    dummy_y_true = np.array([0, 1])
    dummy_y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])  # Perfect predictions
    dummy_loss = calculate_log_loss(dummy_y_true, dummy_y_pred, labels=[0, 1])

    print(f"Dummy Perfect Loss: {dummy_loss:.15f}")
    # With epsilon 1e-15, -log(1 - 1e-15) is approx 1e-15 (effectively 0)
    assert dummy_loss < 1e-10, "Metric calculation failed for perfect predictions"

    # -------------------------------------------------------------------------
    # 7. Prediction & Submission
    # -------------------------------------------------------------------------
    print("\n--- 7. Prediction & Submission ---")

    # Generate predictions for test set
    test_probs = model.predict_proba(X_test_proc)

    # Save submission
    save_submission(test_ids, classes, test_probs, output_path=Config.SUBMISSION_FILE)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    expected_cols = ["id"] + list(classes)
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match requirements"
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
