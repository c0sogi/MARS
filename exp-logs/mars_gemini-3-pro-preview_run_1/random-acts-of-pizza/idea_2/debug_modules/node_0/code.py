import os
import sys
import numpy as np
import pandas as pd
import warnings

# Set random seeds for reproducibility
np.random.seed(42)

# Import library components
from library.config import Config
from library.data_manager import DataManager
from library.feature_extractor import HybridFeatureProcessor
from library.model_wrapper import PizzaRandomForest


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Optimization
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")
    # Modify Config class attributes directly to speed up execution
    Config.RF_N_ESTIMATORS = 10  # Reduce trees for speed
    Config.RF_MAX_DEPTH = 5  # Limit depth
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 rows for processing

    # Ensure working directory exists for temporary outputs
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print("   Configuration updated: N_ESTIMATORS=10, SAMPLE_SIZE=50")

    # -------------------------------------------------------------------------
    # 2. Data Manager Demonstration
    # -------------------------------------------------------------------------
    print("\n2. Demonstrating DataManager...")
    dm = DataManager()

    # Load a small subset of aligned data
    # This aligns schemas (removes leakage columns from train/val)
    train_df, val_df, test_df = dm.get_data(debug_size=Config.DEBUG_SAMPLE_SIZE)

    print(f"   Loaded Train shape: {train_df.shape}")
    print(f"   Loaded Val shape:   {val_df.shape}")
    print(f"   Loaded Test shape:  {test_df.shape}")

    # Validation: Check that leakage columns are removed from Train
    # 'number_of_upvotes_of_request_at_retrieval' is a known leakage column present in raw train
    leakage_col = "number_of_upvotes_of_request_at_retrieval"
    if leakage_col in train_df.columns:
        raise AssertionError(
            f"Leakage column '{leakage_col}' found in aligned training data!"
        )

    # Validation: Check that target exists in Train/Val but not Test
    target_col = "requester_received_pizza"
    assert target_col in train_df.columns, "Target column missing from Train"
    assert target_col in val_df.columns, "Target column missing from Val"
    # Note: align_datasets keeps common features. Target is usually not in test set common features.
    assert target_col not in test_df.columns, "Target column should not be in Test"

    print("   DataManager assertions passed.")

    # -------------------------------------------------------------------------
    # 3. Feature Processor Demonstration
    # -------------------------------------------------------------------------
    print("\n3. Demonstrating HybridFeatureProcessor...")
    fp = HybridFeatureProcessor()

    # Process data: Generate embeddings and engineer features
    # load_cached_data=False forces re-computation to demonstrate logic
    # debug_size is passed to ensure we process the small subset
    print("   Processing data (Text Embeddings + Tabular Features)...")
    train_proc, val_proc, test_proc = fp.process_data(
        load_cached_data=False, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    print(f"   Processed Train shape: {train_proc.shape}")

    # Validation: Check for embedding columns
    # The 'all-MiniLM-L6-v2' model produces 384-dimensional embeddings
    expected_emb_col = "emb_0"
    assert (
        expected_emb_col in train_proc.columns
    ), "Embedding columns missing from processed data"

    # Validation: Check for engineered features
    expected_feat = "feat_upvote_ratio"
    # Note: This feature depends on specific input columns. If they exist in the subset, it should be created.
    # Based on the dataset description, 'requester_upvotes_plus_downvotes_at_request' exists.
    if expected_feat in train_proc.columns:
        print(f"   Engineered feature '{expected_feat}' successfully created.")

    # Validation: Check for missing values in numeric columns (Imputation check)
    # We exclude object columns like request_id
    numeric_cols = train_proc.select_dtypes(include=[np.number]).columns
    assert (
        not train_proc[numeric_cols].isnull().any().any()
    ), "NaNs found in numeric columns after processing"

    print("   HybridFeatureProcessor assertions passed.")

    # -------------------------------------------------------------------------
    # 4. Model Wrapper Demonstration
    # -------------------------------------------------------------------------
    print("\n4. Demonstrating PizzaRandomForest...")
    rf_model = PizzaRandomForest()

    # Prepare Feature Matrices (X) and Target Vectors (y)
    # We need to drop non-feature columns
    drop_cols = [
        "request_id",
        "requester_received_pizza",
        "source_file",
        "request_text",
        "request_title",
    ]
    # Filter out columns that might not exist in processed df (like text source columns if they were kept)
    cols_to_drop_train = [
        c for c in train_proc.columns if c in drop_cols or c == target_col
    ]

    X_train = train_proc.drop(columns=cols_to_drop_train)
    y_train = train_proc[target_col]

    cols_to_drop_val = [
        c for c in val_proc.columns if c in drop_cols or c == target_col
    ]
    X_val = val_proc.drop(columns=cols_to_drop_val)
    y_val = val_proc[target_col]

    # Train the model
    print("   Training model...")
    auc_score = rf_model.train(X_train, y_train, X_val, y_val)
    print(f"   Model trained. Validation AUC: {auc_score:.4f}")

    # Validation: AUC should be a valid float
    assert isinstance(auc_score, float), "AUC score is not a float"
    assert 0 <= auc_score <= 1, "AUC score out of range [0, 1]"

    # Predict on Test Set
    # Align test columns with train columns (ensure order matches)
    # Identify features used in training
    train_features = X_train.columns.tolist()

    # Ensure test set has the same features (fill missing with 0 if any, though alignment should handle this)
    # For this demo, we just select the columns that exist
    X_test = test_proc[train_features]

    print("   Generating predictions...")
    test_probs = rf_model.predict_proba(X_test)

    # Validation: Predictions shape and range
    assert len(test_probs) == len(test_proc), "Prediction length mismatch"
    assert np.all(
        (test_probs >= 0) & (test_probs <= 1)
    ), "Predictions contain values outside [0, 1]"

    print("   Predictions generated successfully.")

    # Feature Importance
    importances = rf_model.get_feature_importance()
    print("\n   Top 3 Features:")
    print(importances.head(3))
    assert not importances.empty, "Feature importances are empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
