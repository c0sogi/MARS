import os
import shutil
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

# Import from the provided library
from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
    TARGET_COL,
    RANDOM_SEED,
)
from library.utils import set_seed, calculate_log_loss, create_submission_file
from library.data_loader import (
    load_datasets,
    get_train_val_data,
    get_full_train_data,
    get_test_data,
    get_feature_columns,
)
from library.transformers import Float64Wrapper, FactorizedDiscriminantProjector
from library.experts import get_expert_library
from library.ensemble import run_selection_phase, _get_input_data


def clean_working_directory():
    """Cleans up the working directory to ensure a fresh run for the demo."""
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"Cleaned working directory: {WORKING_DIR}")


def demo_data_loading():
    print("\n=== Demo: Data Loading & Feature Extraction ===")

    # 1. Load datasets (forces extraction of morphometrics if not cached)
    # We set load_cached_data=False to demonstrate the extraction logic
    print("Loading datasets (this triggers morphometric feature extraction)...")
    df_train, df_val, df_test = load_datasets(load_cached_data=False)

    # Verify shapes
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # Verify content
    expected_morph_cols = ["hu_0", "aspect_ratio", "solidity"]
    for col in expected_morph_cols:
        assert col in df_train.columns, f"Missing expected morphometric column: {col}"

    assert len(df_train) > 0, "Training dataframe is empty"
    assert len(df_val) > 0, "Validation dataframe is empty"
    assert len(df_test) > 0, "Test dataframe is empty"

    print("Data loading and feature extraction verified.")
    return df_train


def demo_transformers(df_sample):
    print("\n=== Demo: Custom Transformers ===")

    # Prepare a small sample
    sample_size = 50
    df_subset = df_sample.head(sample_size).copy()
    X_subset = df_subset[MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS]
    y_subset = df_subset[TARGET_COL]

    # 1. Float64Wrapper
    print("Testing Float64Wrapper...")
    f64_wrapper = Float64Wrapper()
    X_f64 = f64_wrapper.transform(X_subset)
    assert X_f64.dtype == np.float64, "Float64Wrapper did not cast to float64"
    print("Float64Wrapper passed.")

    # 2. FactorizedDiscriminantProjector
    print("Testing FactorizedDiscriminantProjector...")
    n_components = 5
    projector = FactorizedDiscriminantProjector(n_components=n_components)

    # Fit
    projector.fit(X_subset, y_subset)

    # Transform
    X_proj = projector.transform(X_subset)

    # Verify output shape: 3 groups (Margin, Shape, Texture) * n_components
    expected_dim = 3 * n_components
    assert X_proj.shape == (
        sample_size,
        expected_dim,
    ), f"Projector output shape mismatch. Expected ({sample_size}, {expected_dim}), got {X_proj.shape}"

    print(f"Projector output shape: {X_proj.shape}")
    print("FactorizedDiscriminantProjector passed.")


def demo_pipeline_execution():
    print("\n=== Demo: Full Pipeline (Selection & Inference) ===")

    # 1. Get Data Splits
    print("Retrieving Train/Val splits...")
    X_train, y_train, X_val, y_val = get_train_val_data(load_cached_data=True)

    # 2. Get Expert Library
    all_experts = get_expert_library()
    print(f"Total available experts defined: {len(all_experts)}")

    # Optimization: Select a subset of experts to speed up this demo
    # We pick one from each conceptual group (A, B, C)
    demo_experts = [
        all_experts[0],  # Group A: Marginal
        all_experts[6],  # Group A: Robust
        all_experts[9],  # Group B: Morphometric
        all_experts[-1],  # Group C: Interaction
    ]
    print(f"Selected {len(demo_experts)} experts for demonstration.")

    # 3. Run Selection Phase
    # This trains experts on X_train, predicts on X_val, and runs Greedy Selection
    print("Running Selection Phase...")
    selector, label_encoder = run_selection_phase(
        X_train, y_train, X_val, y_val, demo_experts, load_cached_data=False
    )

    # Verify selection
    print(f"Selected weights: {selector.weights}")
    assert len(selector.weights) > 0, "Greedy selector failed to select any experts."

    # 4. Retraining on Full Data (Train + Val)
    print("Retraining selected experts on Full Data (Train + Val)...")
    X_full, y_full = get_full_train_data(load_cached_data=True)
    y_full_enc = label_encoder.transform(y_full)

    # Dictionary to store final test predictions
    test_preds_dict = {}

    # Get Test Data
    X_test, ids_test = get_test_data(load_cached_data=True)

    for expert_name in selector.weights.keys():
        print(f"Retraining {expert_name}...")

        # Find the expert config
        expert_config = next(e for e in demo_experts if e["name"] == expert_name)

        # Prepare data based on input type
        X_full_np = _get_input_data(X_full, expert_config["input_type"])
        X_test_np = _get_input_data(X_test, expert_config["input_type"])

        # Clone and Train
        model = make_pipeline(
            clone(expert_config["pipeline"]), clone(expert_config["estimator"])
        )
        model.fit(X_full_np, y_full_enc)

        # Predict
        test_preds_dict[expert_name] = model.predict_proba(X_test_np)

    # 5. Ensemble Prediction
    print("Generating ensemble predictions...")
    final_probs = selector.predict(test_preds_dict)

    assert final_probs.shape[0] == len(ids_test), "Prediction rows mismatch test IDs"
    assert final_probs.shape[1] == len(
        label_encoder.classes_
    ), "Prediction cols mismatch class count"

    # 6. Submission Generation
    print("Creating submission file...")
    class_names = list(label_encoder.classes_)
    submission_df = create_submission_file(
        ids_test,
        final_probs,
        class_names,
        output_path=os.path.join(SUBMISSION_DIR, "demo_submission.csv"),
    )

    print("Submission file created successfully.")
    print(submission_df.head())


def demo_metrics():
    print("\n=== Demo: Metrics ===")
    # Create dummy data
    y_true = np.array([0, 1, 2])
    # Predictions that don't sum to 1 (to test normalization)
    y_pred = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.0], [0.2, 0.2, 0.6]]  # Sums to 0.9
    )

    loss = calculate_log_loss(y_true, y_pred)
    print(f"Calculated Log Loss: {loss:.4f}")
    assert not np.isnan(loss), "Log loss is NaN"
    assert loss > 0, "Log loss should be positive"


if __name__ == "__main__":
    # Setup
    set_seed(RANDOM_SEED)
    clean_working_directory()

    # Execution Steps
    try:
        # 1. Data Loading
        df_train = demo_data_loading()

        # 2. Transformers
        demo_transformers(df_train)

        # 3. Metrics
        demo_metrics()

        # 4. Full Pipeline
        demo_pipeline_execution()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE IN DEMO: {e}")
        raise e
