import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.utils import seed_everything, get_cache_dir
from library.data_processing import DataProcessor
from library.model_xgb import run_xgb_cv
from library.engine_resnet import run_resnet_cv, ForestDataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Set Seed for Reproducibility
    seed_everything(42)

    # 2. Monkey-Patch DataProcessor to load a subset of data
    # This ensures the entire pipeline runs quickly without processing 3.6M rows.
    print("\n[Step 1] Patching DataProcessor to use data subsets...")

    original_load_raw_data = DataProcessor.load_raw_data

    def mock_load_raw_data(self):
        print("    -> Mock load_raw_data called: Loading subset of metadata...")
        # Load only 2000 rows for training, 500 for validation, 500 for testing
        train_path = os.path.join(self.metadata_dir, "train.csv")
        val_path = os.path.join(self.metadata_dir, "val.csv")
        test_path = os.path.join(self.metadata_dir, "test.csv")

        df_train = pd.read_csv(train_path, nrows=2000)
        df_val = pd.read_csv(val_path, nrows=500)
        df_test = pd.read_csv(test_path, nrows=500)

        return df_train, df_val, df_test

    # Apply the patch to the class
    DataProcessor.load_raw_data = mock_load_raw_data

    # 3. XGBoost Pipeline Demonstration
    print("\n[Step 2] Demonstrating XGBoost Pipeline...")

    # Initialize processor
    processor = DataProcessor()

    # Force regeneration of cache with our subset data
    # load_cached_data=False triggers the (patched) loading and processing logic
    print("    -> Generating XGBoost data (subset)...")
    X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, X_test_xgb, le_xgb = (
        processor.get_xgb_data(load_cached_data=False)
    )

    print(f"    -> XGB Train Data Shape: {X_train_xgb.shape}")

    # Run XGBoost CV
    # We use n_splits=2 for speed
    print("    -> Running XGBoost CV (2 folds)...")
    oof_preds_xgb, test_preds_xgb, le_xgb_out, y_full_xgb = run_xgb_cv(
        load_cached_data=True, n_splits=2, seed=42  # Load the cache we just generated
    )

    # Validation
    expected_train_size = len(X_train_xgb) + len(X_val_xgb)
    expected_test_size = len(X_test_xgb)
    num_classes = len(le_xgb.classes_)

    assert oof_preds_xgb.shape == (
        expected_train_size,
        num_classes,
    ), f"XGB OOF shape mismatch. Expected {(expected_train_size, num_classes)}, got {oof_preds_xgb.shape}"
    assert test_preds_xgb.shape == (
        expected_test_size,
        num_classes,
    ), f"XGB Test Preds shape mismatch. Expected {(expected_test_size, num_classes)}, got {test_preds_xgb.shape}"
    assert len(y_full_xgb) == expected_train_size, "XGB Target length mismatch."

    print("    -> XGBoost Pipeline Verified Successfully.")

    # 4. ResNet Pipeline Demonstration
    print("\n[Step 3] Demonstrating ResNet Pipeline...")

    # Force regeneration of cache for NN (different preprocessing: QuantileTransform, Dense Indices)
    print("    -> Generating ResNet data (subset)...")
    X_train_nn, y_train_nn, X_val_nn, y_val_nn, X_test_nn, le_nn = (
        processor.get_nn_data(load_cached_data=False)
    )

    print(f"    -> NN Train Data Shape: {X_train_nn.shape}")

    # Run ResNet CV
    # epochs=1, n_splits=2, batch_size=128 for speed
    print("    -> Running ResNet CV (2 folds, 1 epoch)...")
    oof_preds_nn, test_preds_nn, le_nn_out, y_full_nn = run_resnet_cv(
        load_cached_data=True, n_splits=2, seed=42, epochs=1, batch_size=128, patience=1
    )

    # Validation
    assert oof_preds_nn.shape == (
        expected_train_size,
        num_classes,
    ), f"ResNet OOF shape mismatch. Expected {(expected_train_size, num_classes)}, got {oof_preds_nn.shape}"
    assert test_preds_nn.shape == (
        expected_test_size,
        num_classes,
    ), f"ResNet Test Preds shape mismatch. Expected {(expected_test_size, num_classes)}, got {test_preds_nn.shape}"

    print("    -> ResNet Pipeline Verified Successfully.")

    # 5. ForestDataset Class Demonstration
    print("\n[Step 4] Verifying ForestDataset (TabularDataset)...")

    # We use the NN data generated above
    # Identify categorical and continuous columns as done in the engine
    cat_cols = ["Soil_Type_Index", "Wilderness_Area_Index"]
    cont_cols = [c for c in X_train_nn.columns if c not in cat_cols]

    # Instantiate Dataset
    ds = ForestDataset(X_train_nn, y_train_nn, cat_cols=cat_cols, cont_cols=cont_cols)

    # Check length
    assert len(ds) == len(X_train_nn), "Dataset length mismatch."

    # Check item retrieval
    x_cat_sample, x_cont_sample, y_sample = ds[0]

    assert isinstance(
        x_cat_sample, torch.Tensor
    ), "Categorical data should be a tensor."
    assert isinstance(
        x_cont_sample, torch.Tensor
    ), "Continuous data should be a tensor."
    assert isinstance(y_sample, torch.Tensor), "Target should be a tensor."

    assert x_cat_sample.shape[0] == len(
        cat_cols
    ), "Categorical feature dimension mismatch."
    assert x_cont_sample.shape[0] == len(
        cont_cols
    ), "Continuous feature dimension mismatch."

    print(
        f"    -> Dataset Sample: Cat={x_cat_sample.shape}, Cont={x_cont_sample.shape}, Target={y_sample}"
    )
    print("    -> ForestDataset Verified Successfully.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
