import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import warnings

# Import library modules
# We import the specific modules where variables might need to be patched for the demo
import library.config
import library.data_loader
import library.embedding_utils
from library.utils import set_seed, get_common_columns
from library.data_loader import load_data
from library.feature_engineering import FeaturePipeline
from library.rf_learner import RFPredictor
from library.mlp_learner import MLPTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Demonstration ===")

    # 1. Setup and Isolation
    # Set seed for reproducibility
    set_seed(42)

    # Create a specific directory for this demo to avoid cache collisions
    demo_dir = os.path.join(library.config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # Monkeypatch WORKING_DIR in relevant modules to redirect cache files
    # This ensures we don't overwrite any full-dataset caches with our subset data
    library.data_loader.WORKING_DIR = demo_dir
    library.embedding_utils.WORKING_DIR = demo_dir

    # Update CACHE_PATHS in config to point to the demo directory
    # Since CACHE_PATHS is a mutable dictionary, updating it here affects FeaturePipeline
    for key in library.config.CACHE_PATHS:
        filename = os.path.basename(library.config.CACHE_PATHS[key])
        library.config.CACHE_PATHS[key] = os.path.join(demo_dir, filename)

    print(f"Working directory set to: {demo_dir}")

    # 2. Data Loading
    print("\n--- Loading and Subsetting Data ---")
    # Load data from source (ignore existing caches to ensure we get raw data)
    train_df, val_df, test_df = load_data(load_cached_data=False)

    # Subset to 50 samples for speed
    subset_size = 50
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    print(f"Data loaded. Subsets created with {subset_size} samples each.")

    # Verify utility function
    common_cols = get_common_columns(train_subset, test_subset)
    assert "request_id" in common_cols
    assert (
        "requester_received_pizza" not in common_cols
    )  # Target should be excluded if not present in test or handled logic

    # 3. Feature Engineering
    print("\n--- Running Feature Pipeline ---")
    pipeline = FeaturePipeline()

    # Execute pipeline
    # load_cached_data=False ensures we compute features for our specific subset
    features = pipeline.fit_transform(
        train_subset, val_subset, test_subset, load_cached_data=False
    )

    # Verify Output Structure
    for split in ["train", "val", "test"]:
        assert split in features, f"Missing {split} in features"
        assert "rf" in features[split], f"Missing RF features for {split}"
        assert "mlp" in features[split], f"Missing MLP features for {split}"

    # Verify RF Features (Sparse)
    X_rf_train = features["train"]["rf"]
    assert sp.issparse(X_rf_train), "RF features should be a sparse matrix"
    assert (
        X_rf_train.shape[0] == subset_size
    ), f"RF feature rows ({X_rf_train.shape[0]}) != subset size"

    # Verify MLP Features (Dict of Arrays)
    X_mlp_train = features["train"]["mlp"]
    assert isinstance(X_mlp_train, dict), "MLP features should be a dictionary"
    assert "title_emb" in X_mlp_train
    assert (
        X_mlp_train["title_emb"].shape[0] == subset_size
    ), "MLP embedding rows mismatch"

    print("Feature pipeline completed and verified.")

    # 4. Random Forest Model
    print("\n--- Training Random Forest ---")
    # Prepare targets
    y_train = train_subset["requester_received_pizza"].values
    y_val = val_subset["requester_received_pizza"].values

    # Configure for speed
    rf_params = {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": 0,
    }

    rf_model = RFPredictor(params=rf_params)

    # Train
    rf_model.train(X_rf_train, y_train, features["val"]["rf"], y_val)

    # Predict
    rf_preds = rf_model.predict(features["test"]["rf"])

    # Validation
    assert len(rf_preds) == subset_size
    assert np.all(
        (rf_preds >= 0) & (rf_preds <= 1)
    ), "RF predictions out of [0,1] range"
    print(f"RF Predictions generated. Mean probability: {np.mean(rf_preds):.4f}")

    # 5. MLP Model
    print("\n--- Training MLP ---")
    # Configure for speed (minimal epochs/batch size)
    mlp_params = {
        "input_embedding_dim": 384,  # Matches SBERT dimension
        "hidden_dim": 32,
        "dropout_prob": 0.0,
        "dropout_dense": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": 2,
        "patience": 1,
        "scheduler_factor": 0.5,
        "scheduler_patience": 1,
    }

    mlp_trainer = MLPTrainer(params=mlp_params)

    # Train
    # Note: We pass numpy arrays for targets; the internal DataLoader handles conversion to Tensor
    mlp_trainer.train(
        X_train=features["train"]["mlp"],
        y_train=y_train,
        X_val=features["val"]["mlp"],
        y_val=y_val,
    )

    # Predict
    mlp_preds = mlp_trainer.predict(features["test"]["mlp"])

    # Validation
    assert len(mlp_preds) == subset_size
    assert np.all(
        (mlp_preds >= 0) & (mlp_preds <= 1)
    ), "MLP predictions out of [0,1] range"
    print(f"MLP Predictions generated. Mean probability: {np.mean(mlp_preds):.4f}")

    # 6. Ensemble (Demonstration)
    print("\n--- Ensemble Demonstration ---")
    ensemble_preds = (rf_preds + mlp_preds) / 2
    print(f"Ensemble predictions generated. First 5: {ensemble_preds[:5]}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
