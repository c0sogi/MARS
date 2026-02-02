import os
import sys
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.ensemble import BaggingClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from the provided library
from library.utils import set_seed, setup_logger
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    STRUCTURAL_COLS,
    TEMPORAL_COLS,
    USER_METADATA_COLS,
    SEMANTIC_TEXT_COLS,
)
from library.data_loader import get_dataset
from library.feature_engine import generate_features, ScalerWrapper
from library.model_factory import (
    create_bagged_logistic_ensemble,
    get_hyperparameter_grid,
)
from library.runner import run_cv_training, generate_submission


def main():
    print("Starting Library Usage Demonstration...")

    # 1. Setup and Seeding
    print("\n[1] Setting up environment and seeds...")
    set_seed(42)
    logger = setup_logger("demo_logger", level=20)  # INFO level
    print("Seed set and logger initialized.")

    # 2. Data Loading Demonstration
    print("\n[2] Demonstrating Data Loading (Debug Mode)...")
    # Load a subset of training data (100 samples)
    df_train_raw = get_dataset("train", debug=True, debug_size=100)

    # Validation
    assert isinstance(
        df_train_raw, pd.DataFrame
    ), "Returned object should be a DataFrame"
    assert (
        len(df_train_raw) == 100
    ), f"Expected 100 rows in debug mode, got {len(df_train_raw)}"
    assert "request_id" in df_train_raw.columns, "request_id column missing"
    assert (
        "requester_received_pizza" in df_train_raw.columns
    ), "Target column missing in train set"
    print("Data loading verified successfully.")

    # 3. Feature Engineering Demonstration
    print("\n[3] Demonstrating Feature Engineering...")
    # Generate features for the training subset
    # This involves text embedding (Transformer), structural extraction, etc.
    print("Generating features (this may take a moment to load the transformer)...")
    df_features = generate_features(
        "train", load_cached_data=False, debug=True, debug_size=50
    )

    # Validation of generated columns
    print("Verifying feature columns...")
    # Check Semantic Embeddings (View 1) - should have 'emb_0', 'emb_1', etc.
    assert any(
        c.startswith("emb_") for c in df_features.columns
    ), "Semantic embeddings missing"

    # Check Structural Features (View 2)
    for col in STRUCTURAL_COLS:
        assert col in df_features.columns, f"Structural feature {col} missing"

    # Check Temporal Features (View 3)
    for col in TEMPORAL_COLS:
        assert col in df_features.columns, f"Temporal feature {col} missing"

    # Check User Metadata (View 4)
    for col in USER_METADATA_COLS:
        assert col in df_features.columns, f"User metadata {col} missing"

    # Check Target
    assert (
        "requester_received_pizza" in df_features.columns
    ), "Target column lost during feature generation"
    print(f"Feature generation verified. Shape: {df_features.shape}")

    # 4. Scaler Demonstration (RankGauss)
    print("\n[4] Demonstrating ScalerWrapper (RankGauss)...")
    cols_to_scale = STRUCTURAL_COLS + TEMPORAL_COLS
    scaler = ScalerWrapper(columns_to_scale=cols_to_scale)

    # Fit on the generated features
    scaler.fit(df_features)

    # Transform
    df_scaled = scaler.transform(df_features)

    # Validate transformation
    # RankGauss with output_distribution='normal' should change values
    # We check one column to ensure it's not identical to original
    check_col = STRUCTURAL_COLS[0]
    original_vals = df_features[check_col].values
    scaled_vals = df_scaled[check_col].values

    assert not np.array_equal(
        original_vals, scaled_vals
    ), "Scaler did not transform the data"
    assert df_scaled.shape == df_features.shape, "Scaler changed DataFrame shape"
    print("ScalerWrapper logic verified.")

    # 5. Model Factory Demonstration
    print("\n[5] Demonstrating Model Factory...")
    model = create_bagged_logistic_ensemble(n_estimators=5, random_state=42)

    assert isinstance(
        model, BaggingClassifier
    ), "Factory did not return a BaggingClassifier"
    assert model.n_estimators == 5, "Model parameter n_estimators mismatch"

    grid = get_hyperparameter_grid()
    assert isinstance(grid, dict), "Hyperparameter grid should be a dictionary"
    assert (
        "estimator__C" in grid
    ), "Grid missing logistic regression 'C' parameter mapping"
    print("Model factory and grid generation verified.")

    # 6. Full Pipeline Execution (Runner)
    print("\n[6] Demonstrating Full CV Training Pipeline (Debug Mode)...")
    # This runs the 5-fold CV on the debug subset (100 samples)
    # It performs feature generation, scaling, grid search, and evaluation
    mean_auc = run_cv_training(debug=True, load_cached_features=False)

    assert isinstance(mean_auc, float), "CV Training should return a float score"
    assert 0 <= mean_auc <= 1, f"AUC score out of bounds: {mean_auc}"
    print(f"CV Training completed. Mean AUC: {mean_auc:.4f}")

    # Check if model artifacts were saved
    fold_0_model_path = os.path.join(WORKING_DIR, "models", "model_fold_0.joblib")
    assert os.path.exists(fold_0_model_path), "Model artifact for fold 0 not found"
    print("Model artifacts verified.")

    # 7. Submission Generation
    print("\n[7] Demonstrating Submission Generation (Debug Mode)...")
    # Generates predictions for the test set (debug subset)
    generate_submission(debug=True, load_cached_features=False)

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert "request_id" in df_sub.columns, "Submission missing request_id"
    assert (
        "requester_received_pizza" in df_sub.columns
    ), "Submission missing probability column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check probabilities are valid
    probs = df_sub["requester_received_pizza"]
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of [0, 1] range"

    print(f"Submission generated successfully at {SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(df_sub.head())

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
