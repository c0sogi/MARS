import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import library modules
# We import config first to override settings for the demo
import library.config as config
from library.utils import set_seed, setup_logger
from library.data_loader import (
    load_labeled_data,
    load_test_data,
    extract_text_data,
    extract_numeric_data,
)
from library.feature_engineering import generate_embeddings, prepare_design_matrix
from library.model_factory import build_ensemble_pipeline
from library.trainer import tune_component


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print(">>> [1/6] Setting up configuration for demonstration...")

    # Override config for speed
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    config.N_JOBS = 1
    config.N_FOLDS = 2

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Setup logger
    logger = setup_logger(os.path.join(config.WORKING_DIR, "demo_execution.log"))
    logger.info("Starting demo execution...")

    # ==========================================
    # 2. Data Loading (Debug Mode)
    # ==========================================
    print(">>> [2/6] Loading labeled data (Debug Mode)...")

    # Load labeled data (train + val merged)
    # This uses the debug flag to load only DEBUG_SAMPLE_SIZE rows
    df_train = load_labeled_data(load_cached_data=False, debug=True)

    # Validation
    assert isinstance(df_train, pd.DataFrame), "Loaded data is not a DataFrame"
    assert (
        len(df_train) <= config.DEBUG_SAMPLE_SIZE
    ), f"Data size {len(df_train)} exceeds debug limit"
    assert "requester_received_pizza" in df_train.columns, "Target column missing"

    logger.info(f"Loaded {len(df_train)} training samples.")

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    print(">>> [3/6] Generating features...")

    # Extract Text
    text_data = extract_text_data(df_train)
    assert len(text_data) == len(df_train), "Text data length mismatch"

    # Generate Embeddings (using a small batch size for the demo)
    # We use a unique cache name 'demo_train' to avoid overwriting real training cache
    embeddings = generate_embeddings(
        text_data, "demo_train", load_cached_data=False, batch_size=16
    )

    # Extract Numeric Data
    numeric_data = extract_numeric_data(df_train)

    # Combine into Design Matrix
    X, metadata_start_idx = prepare_design_matrix(embeddings, numeric_data)
    y = df_train["requester_received_pizza"].values.astype(int)

    # Validation
    n_samples, n_features = X.shape
    embedding_dim = embeddings.shape[1]
    numeric_dim = numeric_data.shape[1]

    assert n_samples == len(df_train), "Design matrix row count mismatch"
    assert n_features == embedding_dim + numeric_dim, "Feature count mismatch"
    assert metadata_start_idx == embedding_dim, "Metadata start index incorrect"

    logger.info(f"Design Matrix Shape: {X.shape}")
    logger.info(f"Metadata starts at index: {metadata_start_idx}")

    # ==========================================
    # 4. Model Component Tuning
    # ==========================================
    print(">>> [4/6] Demonstrating component tuning...")

    # Create a simple train/val split for the demo
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )

    # Get a dummy pipeline to extract the preprocessor
    dummy_pipeline = build_ensemble_pipeline(metadata_start_idx)
    preprocessor = dummy_pipeline.named_steps["preprocessor"]

    # Define a tiny grid for Logistic Regression to prove it runs
    tiny_lr_grid = {"C": [0.1, 1.0], "solver": ["liblinear"]}

    # Run tuning
    logger.info("Tuning LR component with tiny grid...")
    best_params, best_score = tune_component(
        X_train, y_train, preprocessor, "lr", tiny_lr_grid
    )

    # Validation
    assert isinstance(best_params, dict), "Best params should be a dictionary"
    assert "C" in best_params, "Expected parameter 'C' in best_params"
    assert isinstance(best_score, float), "Score should be a float"

    logger.info(f"Tuning complete. Best Params: {best_params}, Score: {best_score:.4f}")

    # ==========================================
    # 5. Pipeline Construction & Training
    # ==========================================
    print(">>> [5/6] Building and training full ensemble pipeline...")

    # Build pipeline with the tuned parameters (and defaults for others)
    pipeline = build_ensemble_pipeline(
        metadata_start_idx, lr_params={**best_params, "random_state": config.SEED}
    )

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Predict on validation set
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

    # Calculate Score
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
        logger.info(f"Validation AUC on demo split: {auc:.4f}")
    except ValueError:
        # Handle case where y_val might have only one class in extremely small debug splits
        logger.warning(
            "Could not calculate AUC (possibly only one class in validation split)."
        )

    # Save the model temporarily
    model_path = os.path.join(config.WORKING_DIR, "demo_model.joblib")
    joblib.dump(pipeline, model_path)
    assert os.path.exists(model_path), "Model file was not saved"

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print(">>> [6/6] Running inference on test data...")

    # Load Test Data (Debug Mode)
    df_test = load_test_data(load_cached_data=False, debug=True)
    assert len(df_test) > 0, "Test data is empty"

    # Process Test Features
    test_text = extract_text_data(df_test)
    test_embeddings = generate_embeddings(
        test_text, "demo_test", load_cached_data=False, batch_size=16
    )
    test_numeric = extract_numeric_data(df_test)
    X_test, _ = prepare_design_matrix(test_embeddings, test_numeric)

    # Load Model
    loaded_pipeline = joblib.load(model_path)

    # Predict
    test_preds = loaded_pipeline.predict_proba(X_test)[:, 1]

    # Validation
    assert len(test_preds) == len(df_test), "Prediction count mismatch"
    assert np.all(
        (test_preds >= 0) & (test_preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": test_preds}
    )

    logger.info("Sample Predictions:")
    print(submission.head())

    print("\n>>> Demo Execution Completed Successfully!")


if __name__ == "__main__":
    main()
