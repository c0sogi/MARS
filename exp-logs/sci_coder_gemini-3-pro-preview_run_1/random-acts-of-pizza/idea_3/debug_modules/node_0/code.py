import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import warnings
import torch

# Import provided library modules
import library.config
import library.data_processing
import library.feature_streams
import library.ensemble_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demo...")

    # 1. Setup Temporary Environment and Optimize for Speed
    # We use a temporary directory for caching to avoid interfering with the main working directory
    # and to demonstrate the caching logic on a fresh run.
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary working directory: {temp_dir}")

    try:
        # Patch the configuration to use the temp directory
        library.config.CACHE_DIR = temp_dir
        # We must also patch the imported variables in the other modules since they were bound at import time
        library.data_processing.CACHE_DIR = temp_dir
        library.feature_streams.CACHE_DIR = temp_dir

        # Reduce model hyperparameters for rapid execution
        library.ensemble_model.RF_ESTIMATORS = 10  # Reduced from 300
        library.ensemble_model.LR_MAX_ITER = 20  # Reduced from 1000

        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)

        # 2. Load and Preprocess Data
        print("\n--- Step 1: Loading Data ---")
        # load_cached_data=True will look in our empty temp_dir, fail to find files, and load from CSVs
        train_df, val_df, test_df, feature_cols = library.data_processing.load_data(
            load_cached_data=True
        )

        # Verify data loading
        assert not train_df.empty, "Training dataframe is empty"
        assert not val_df.empty, "Validation dataframe is empty"
        assert not test_df.empty, "Test dataframe is empty"
        print(f"Original Train shape: {train_df.shape}")

        # Subsample data for speed (Top 50 samples each)
        print("Subsampling data to 50 rows each for speed...")
        train_df = train_df.head(50).reset_index(drop=True)
        val_df = val_df.head(50).reset_index(drop=True)
        test_df = test_df.head(50).reset_index(drop=True)

        # 3. Generate Feature Streams
        print("\n--- Step 2: Generating Feature Streams ---")
        # This will trigger the SparseStreamTransformer and DenseStreamTransformer
        # Since we are using a fresh temp_dir, this will compute features and save them to the temp cache
        sparse_data, dense_data = library.feature_streams.generate_streams(
            train_df, val_df, test_df, feature_cols, load_cached_data=True
        )

        # Verify Sparse Data (Dictionary of sparse matrices)
        assert "train" in sparse_data
        assert sparse_data["train"].shape[0] == 50
        print(f"Sparse Train Shape: {sparse_data['train'].shape}")

        # Verify Dense Data (Dictionary of numpy arrays)
        assert "train" in dense_data
        assert dense_data["train"].shape[0] == 50
        print(f"Dense Train Shape: {dense_data['train'].shape}")

        # 4. Train Hybrid Ensemble Model
        print("\n--- Step 3: Training Hybrid Ensemble ---")
        # Prepare target variables
        y_train = train_df["requester_received_pizza"].values
        y_val = val_df["requester_received_pizza"].values

        # Instantiate model
        model = library.ensemble_model.HybridEnsemble(
            rf_estimators=library.ensemble_model.RF_ESTIMATORS,
            lr_max_iter=library.ensemble_model.LR_MAX_ITER,
        )

        # Fit model
        model.fit(
            X_sparse_train=sparse_data["train"],
            X_dense_train=dense_data["train"],
            y_train=y_train,
            X_sparse_val=sparse_data["val"],
            X_dense_val=dense_data["val"],
            y_val=y_val,
        )
        print("Model training complete.")

        # 5. Inference and Validation
        print("\n--- Step 4: Inference ---")
        # Predict on test set
        test_probs = model.predict_proba(sparse_data["test"], dense_data["test"])

        # Verify predictions
        assert len(test_probs) == 50, "Prediction count mismatch"
        assert np.all(
            (test_probs >= 0) & (test_probs <= 1)
        ), "Probabilities out of bounds [0, 1]"

        print(f"Generated {len(test_probs)} predictions.")
        print(f"First 5 predictions: {test_probs[:5]}")
        print("Logic verification passed successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise e
    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    run_demo()
