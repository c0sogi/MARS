import os
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_datasets
from library.feature_engineering import FusionTransformer
from library.model import get_bagged_ensemble, tune_ensemble_hyperparameters
from library.pipeline import run_stratified_cv


def run_demo():
    print("=== Starting Demonstration of Library Components ===")

    # ------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    set_seed(42)

    # Override Config attributes to ensure the demo runs quickly and handles small data
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.N_FOLDS = 2  # Reduce folds from 5 to 2
    Config.N_BAGGING_ESTIMATORS = 2  # Reduce ensemble size from 20 to 2
    Config.PCA_COMPONENTS = 5  # Reduce PCA components (must be < n_samples=10)

    # Simplify Grid Search to a single iteration
    Config.GRID_PARAMS = {"C": [1.0], "class_weight": [None]}

    # Clean up previous demo runs to ensure fresh execution
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 2. Demonstrate Data Loading (Embedder + Metadata)
    # ------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading and Embedding Generation...")
    # Use a very small sample size (10) to minimize runtime of BERT inference
    debug_size = 10

    # load_datasets handles loading JSONs, merging metadata, and generating embeddings
    data = load_datasets(load_cached_data=False, debug_sample_size=debug_size)

    # Verify Data Structure
    required_keys = [
        "df_train",
        "df_test",
        "y_train",
        "meta_train",
        "meta_test",
        "train_embeddings",
        "test_embeddings",
    ]
    for key in required_keys:
        assert key in data, f"Missing key in data dictionary: {key}"

    # Verify Shapes
    n_samples = debug_size
    assert len(data["df_train"]) == n_samples
    assert len(data["y_train"]) == n_samples

    # Verify Embeddings (Anchor: 384d, Aux1/2: 768d)
    embeddings = data["train_embeddings"]
    assert embeddings["anchor"].shape == (n_samples, 384)
    assert embeddings["aux1"].shape == (n_samples, 768)
    assert embeddings["aux2"].shape == (n_samples, 768)

    # Verify Metadata (10 features as per config)
    assert data["meta_train"].shape == (n_samples, 10)

    print(
        f"Successfully loaded {debug_size} samples and generated multi-view embeddings."
    )

    # ------------------------------------------------------------------------
    # 3. Demonstrate Feature Engineering (FusionTransformer)
    # ------------------------------------------------------------------------
    print("\n[3] Demonstrating Feature Engineering (FusionTransformer)...")

    # Prepare dictionary for transformer (simulating a fold split)
    X_train_dict = {
        "anchor": data["train_embeddings"]["anchor"],
        "aux1": data["train_embeddings"]["aux1"],
        "aux2": data["train_embeddings"]["aux2"],
        "meta": data["meta_train"],
    }

    # Instantiate Transformer
    transformer = FusionTransformer()

    # Fit (learns PCA and Quantile Scaling)
    transformer.fit(X_train_dict)

    # Transform (Applies projections and concatenation)
    X_fused = transformer.transform(X_train_dict)

    # Calculate expected dimensions:
    # Anchor (384) + Aux1_PCA (5) + Aux2_PCA (5) + Meta (10) = 404
    expected_dim = 384 + Config.PCA_COMPONENTS + Config.PCA_COMPONENTS + 10

    print(f"Fused Feature Shape: {X_fused.shape}")
    assert X_fused.shape == (
        n_samples,
        expected_dim,
    ), f"Expected shape ({n_samples}, {expected_dim}), got {X_fused.shape}"

    # Ensure no NaNs
    assert not np.isnan(X_fused).any(), "Fused features contain NaNs"

    print("FusionTransformer logic verified.")

    # ------------------------------------------------------------------------
    # 4. Demonstrate Model Training and Tuning
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Training and Tuning...")

    y_train = data["y_train"]

    # Use the same data for validation in this demo
    X_val = X_fused
    y_val = y_train

    # Test basic model instantiation and prediction
    model = get_bagged_ensemble(n_estimators=Config.N_BAGGING_ESTIMATORS)
    model.fit(X_fused, y_train)
    probs = model.predict_proba(X_fused)[:, 1]

    assert probs.shape == (n_samples,)
    assert (probs >= 0).all() and (probs <= 1).all()

    # Test Hyperparameter Tuning Wrapper
    print("Running hyperparameter tuning...")
    best_model, best_params, best_score = tune_ensemble_hyperparameters(
        X_train=X_fused,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=Config.GRID_PARAMS,
        n_estimators=Config.N_BAGGING_ESTIMATORS,
        random_state=42,
    )

    print(f"Best Validation AUC: {best_score:.4f}")
    assert best_model is not None
    assert isinstance(best_params, dict)

    print("Model training and tuning verified.")

    # ------------------------------------------------------------------------
    # 5. Demonstrate Full Pipeline Execution
    # ------------------------------------------------------------------------
    print("\n[5] Demonstrating Full Pipeline Execution...")

    # run_stratified_cv integrates data loading, feature engineering, CV, and submission
    # We use the debug_sample_size to keep it fast
    run_stratified_cv(debug_sample_size=debug_size, load_cached_data=False)

    # Verify Submission Output
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created at: {submission_path}")
    print(f"Submission shape: {df_sub.shape}")

    # Check if submission has correct number of rows (debug_size)
    assert len(df_sub) == debug_size
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
