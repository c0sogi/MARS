import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_engineering as feature_engineering
import library.model as model_lib
import library.trainer as trainer


def run_demonstration():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup and Configuration Override for Speed
    # We modify the config module directly to optimize for a quick demo run.
    print("\n[1] Configuring environment for rapid demonstration...")
    utils.set_seed(42)

    # Override config values to run on a tiny subset with reduced complexity
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 rows
    config.MAX_ITER = 10  # Limit solver iterations
    config.MAX_FEATURES_WORD = 500  # Reduce vectorizer dimensionality
    config.MAX_FEATURES_CHAR = 500
    config.C_PARAM = 0.1  # Weaker regularization for quick convergence test

    # Force cache regeneration for this demo by using unique paths or disabling loading
    # We will simply pass load_cached_data=False to functions.

    print(f"Debug Mode: {config.DEBUG}")
    print(f"Sample Size: {config.DEBUG_SAMPLE_SIZE}")

    # 2. Verify Utility Functions (Metric)
    print("\n[2] Verifying Metric Calculation (utils.py)...")
    # Test Case: Perfect prediction
    y_true_perf = ["EAP", "HPL"]
    # Probabilities: EAP, HPL, MWS
    y_pred_perf = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    loss_perf = utils.calculate_log_loss(y_true_perf, y_pred_perf)
    print(f"Perfect Prediction Log Loss: {loss_perf}")

    # Due to clipping (1e-15), loss should be very close to 0 but not exactly 0
    assert loss_perf < 1e-5, "Perfect prediction should have near-zero log loss."

    # Test Case: Rescaling logic (probabilities don't sum to 1)
    y_true_scale = ["EAP"]
    y_pred_scale = [[0.5, 0.5, 0.5]]  # Sums to 1.5
    # Should be rescaled to [0.33, 0.33, 0.33]
    loss_scale = utils.calculate_log_loss(y_true_scale, y_pred_scale)
    expected_prob = 1.0 / 3.0
    expected_loss = -np.log(expected_prob)
    print(f"Rescaling Test Loss: {loss_scale} (Expected approx {expected_loss})")
    assert np.isclose(loss_scale, expected_loss, atol=1e-5), "Rescaling logic failed."

    # 3. Verify Data Loading
    print("\n[3] Verifying Data Loading (data_loader.py)...")
    train_df, val_df, test_df = data_loader.load_and_preprocess_data(
        load_cached_data=False, nrows=config.DEBUG_SAMPLE_SIZE
    )

    print(f"Train Rows: {len(train_df)}")
    print(f"Val Rows: {len(val_df)}")

    assert len(train_df) == config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch."
    assert len(val_df) == config.DEBUG_SAMPLE_SIZE, "Val DF size mismatch."
    assert "text" in train_df.columns, "Text column missing."
    # Check preprocessing (lowercase)
    if config.LOWERCASE:
        sample_text = train_df.iloc[0]["text"]
        assert (
            sample_text == sample_text.lower()
        ), "Text preprocessing (lowercase) failed."

    # 4. Verify Feature Engineering
    print("\n[4] Verifying Feature Engineering (feature_engineering.py)...")
    # This function fits vectorizers and transforms data
    X_train, y_train, X_val, y_val, X_test = feature_engineering.extract_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Check shapes
    assert X_train.shape[0] == config.DEBUG_SAMPLE_SIZE
    assert scipy.sparse.issparse(X_train), "X_train should be a sparse matrix."

    # Check label encoding (should be integers 0, 1, 2)
    unique_labels = np.unique(y_train)
    assert np.all(
        np.isin(unique_labels, [0, 1, 2])
    ), "Labels should be encoded as integers."

    # 5. Verify Model Training
    print("\n[5] Verifying Model Training (model.py)...")

    # Convert integer labels back to strings for the model wrapper,
    # as the provided library expects string labels to align with config.CLASSES
    classes_arr = np.array(config.CLASSES)
    y_train_str = classes_arr[y_train]
    y_val_str = classes_arr[y_val]

    # Build model
    model = model_lib.build_model()
    assert isinstance(model, LogisticRegression), "Model should be LogisticRegression."
    assert (
        model.max_iter == config.MAX_ITER
    ), "Config override for max_iter not applied."

    # Train model
    model, metrics = model_lib.train_model(
        model, X_train, y_train_str, X_val, y_val_str
    )

    print(f"Training Metrics: {metrics}")
    assert "log_loss" in metrics, "Metrics should contain log_loss."
    assert hasattr(model, "classes_"), "Model should be fitted."

    # Verify classes alignment
    assert np.array_equal(
        model.classes_, config.CLASSES
    ), "Model classes must match config classes to ensure prediction column order."

    # 6. Verify Full Pipeline (Trainer)
    print("\n[6] Verifying Full Pipeline (trainer.py)...")
    # This runs loading -> extraction -> training -> prediction -> submission
    # We use debug=True to use the sample size defined in config
    trained_model = trainer.run_training(debug=True, load_cached_data=False)

    assert trained_model is not None, "Trainer should return a model."

    # Check if submission file was created
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(submission_df.head())

    assert "id" in submission_df.columns, "Submission missing 'id' column."
    for cls in config.CLASSES:
        assert cls in submission_df.columns, f"Submission missing class column '{cls}'."

    # Check row count (should match test set size, which is DEBUG_SAMPLE_SIZE in this run)
    assert (
        len(submission_df) == config.DEBUG_SAMPLE_SIZE
    ), "Submission row count mismatch."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
