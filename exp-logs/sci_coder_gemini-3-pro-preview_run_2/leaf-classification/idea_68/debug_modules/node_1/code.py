import os
import importlib
import numpy as np
import pandas as pd
import warnings

# Reload config and data_handler to ensure column definitions are updated in persistent environment
import library.config
import library.data_handler

importlib.reload(library.config)
importlib.reload(library.data_handler)

# Import library components
from library.utils import set_seed, save_submission, compute_log_loss
from library.data_handler import get_data
from library.pipeline_definitions import (
    get_global_pipeline,
    get_stratified_rotational_pipeline,
    get_morphometric_pipeline,
)
from library.ensemble_strategy import GreedySelector, aggregate_predictions
from library.config import SUBMISSION_PATH

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    # 1. Setup and Initialization
    print("Initializing demonstration...")
    set_seed(42)

    # 2. Data Loading
    # Uses data_handler to load merged features from cache or metadata
    print("Loading data...")
    (
        X_train_dict,
        y_train,
        X_val_dict,
        y_val,
        X_test_dict,
        test_ids,
        classes,
    ) = get_data(load_cached_data=True)

    # Verify data integrity
    assert len(X_train_dict["Global"]) == len(y_train), "Train features/labels mismatch"
    assert len(X_val_dict["Global"]) == len(y_val), "Val features/labels mismatch"
    print(
        f"Data loaded: {len(y_train)} Train, {len(y_val)} Val, {len(test_ids)} Test samples."
    )
    print(f"Number of classes: {len(classes)}")

    # 3. Model Definition and Training
    # We select a subset of pipelines to demonstrate the library's capabilities.
    # Format: "ModelName": (PipelineFactoryFunction, ShrinkageParam, FeatureViewKey)
    model_configs = {
        "Global_LDA_Auto": (get_global_pipeline, "auto", "Global"),
        "Global_LDA_0.1": (get_global_pipeline, 0.1, "Global"),
        "Stratified_Rot_LDA_Auto": (
            get_stratified_rotational_pipeline,
            "auto",
            "Global",
        ),
        "Morphometric_LDA_Auto": (get_morphometric_pipeline, "auto", "Morphometrics"),
    }

    val_predictions = {}
    test_predictions = {}

    print("\nTraining models and generating predictions...")
    for name, (pipeline_func, shrinkage, feature_key) in model_configs.items():
        print(f"  Processing {name}...")

        # Retrieve the correct feature view (e.g., Global 192 features or Morphometric 11 features)
        X_train = X_train_dict[feature_key]
        X_val = X_val_dict[feature_key]
        X_test = X_test_dict[feature_key]

        # Initialize pipeline
        pipeline = pipeline_func(shrinkage)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict (Probabilities)
        p_val = pipeline.predict_proba(X_val)
        p_test = pipeline.predict_proba(X_test)

        # Store predictions
        val_predictions[name] = p_val
        test_predictions[name] = p_test

        # Evaluate individual performance
        loss = compute_log_loss(y_val, p_val, classes=classes)
        print(f"    -> Validation Log Loss: {loss:.5f}")

    # 4. Ensemble Optimization
    # Use Greedy Forward Selection to find optimal weights for the models
    print("\nOptimizing ensemble weights (Greedy Forward Selection)...")
    selector = GreedySelector(n_iterations=20, random_state=42)
    selector.fit(val_predictions, y_val, classes)

    best_weights = selector.get_best_weights()
    print(f"Optimal Weights: {best_weights}")

    # 5. Prediction Aggregation
    print("Aggregating test predictions...")
    final_test_probs = aggregate_predictions(test_predictions, best_weights)

    # 6. Submission Generation
    print(f"Saving submission to {SUBMISSION_PATH}...")
    save_submission(test_ids, classes, final_test_probs, output_path=SUBMISSION_PATH)

    # 7. Verification and Logic Checks
    print("\nVerifying submission integrity...")

    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(SUBMISSION_PATH)

    # Check 1: Shape
    expected_rows = len(test_ids)
    expected_cols = len(classes) + 1  # +1 for 'id' column
    if df_sub.shape != (expected_rows, expected_cols):
        raise AssertionError(
            f"Submission shape mismatch. Expected {(expected_rows, expected_cols)}, got {df_sub.shape}"
        )

    # Check 2: ID alignment
    if not np.array_equal(df_sub["id"].values, test_ids):
        raise AssertionError("Submission IDs do not match test set IDs.")

    # Check 3: Probability constraints
    probs = df_sub.drop(columns=["id"]).values
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Probabilities found outside [0, 1] range.")

    # Check 4: Row Sums (Should be ~1.0, allowing for small epsilon from clipping)
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-2):
        raise AssertionError(
            f"Probabilities do not sum to 1. Max deviation: {np.abs(row_sums - 1.0).max()}"
        )

    print("Verification passed. Demonstration complete.")


if __name__ == "__main__":
    run_demonstration()
