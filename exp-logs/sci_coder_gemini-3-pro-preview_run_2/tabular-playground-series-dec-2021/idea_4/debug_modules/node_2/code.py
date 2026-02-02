import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.feature_engineering import process_data
from library.data_factory import load_data, prepare_nn_data, get_cv_folds, ForestDataset
from library.xgb_trainer import train_xgb_fold, predict_xgb
from library.nn_trainer import train_nn_fold, predict_nn
from library.ensemble_optimizer import optimize_blending_weights, weighted_average


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing Demonstration Script...")
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("Overriding configuration for fast demonstration...")
    # XGBoost: Reduce estimators and depth for quick training
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3
    Config.XGB_PARAMS["early_stopping_rounds"] = 5

    # Neural Network: Reduce epochs and batch size logic
    Config.NN_PARAMS["epochs"] = 2
    Config.NN_PARAMS["hidden_layers"] = [64, 32]  # Smaller net for demo
    Config.NN_PARAMS["batch_size"] = 256

    # ---------------------------------------------------------
    # 2. Data Loading & Feature Engineering
    # ---------------------------------------------------------
    print("\n--- Step 2: Data Loading (Tabular) ---")
    # This loads the full dataset and applies feature engineering (interactions)
    # It caches the result in ./working/idea_4/ to speed up subsequent runs
    X_train_full, y_train_full, X_test_full, test_ids = load_data(load_cached_data=True)

    print(f"Full Training Data Shape: {X_train_full.shape}")
    print(f"Full Test Data Shape: {X_test_full.shape}")

    # Subsample for demonstration speed (e.g., 5000 samples)
    # We use a fixed subset to simulate a smaller dataset
    demo_size = 5000

    # Ensure all classes are present to prevent XGBoost errors about missing classes
    unique_classes = np.unique(y_train_full)
    forced_indices = [
        np.random.choice(np.where(y_train_full == c)[0]) for c in unique_classes
    ]

    # Fill the rest randomly
    remaining_count = demo_size - len(forced_indices)
    random_indices = np.random.choice(len(X_train_full), remaining_count, replace=False)

    indices = np.concatenate([forced_indices, random_indices])
    np.random.shuffle(indices)

    X_train_demo = X_train_full.iloc[indices].reset_index(drop=True)
    y_train_demo = y_train_full[indices]

    print(f"Subsampled Demo Data Shape: {X_train_demo.shape}")

    # ---------------------------------------------------------
    # 3. XGBoost Training & Prediction
    # ---------------------------------------------------------
    print("\n--- Step 3: XGBoost Workflow ---")

    # Create a simple validation split from the demo data
    val_size = 1000
    X_tr_xgb = X_train_demo.iloc[:-val_size]
    y_tr_xgb = y_train_demo[:-val_size]
    X_val_xgb = X_train_demo.iloc[-val_size:]
    y_val_xgb = y_train_demo[-val_size:]

    print("Training XGBoost on demo split...")
    xgb_model = train_xgb_fold(X_tr_xgb, y_tr_xgb, X_val_xgb, y_val_xgb)

    # Validate Logic: Check model type
    assert xgb_model is not None, "XGBoost model training failed (returned None)"

    # Predict on Validation set
    xgb_val_preds = predict_xgb(xgb_model, X_val_xgb)
    print(f"XGBoost Validation Predictions Shape: {xgb_val_preds.shape}")
    assert xgb_val_preds.shape == (
        val_size,
        Config.NUM_CLASSES,
    ), "XGBoost prediction shape mismatch"

    # Predict on a subset of Test set
    X_test_subset = X_test_full.iloc[:100]
    xgb_test_preds = predict_xgb(xgb_model, X_test_subset)
    print("XGBoost Test Predictions generated.")

    # ---------------------------------------------------------
    # 4. Neural Network Data Prep & Training
    # ---------------------------------------------------------
    print("\n--- Step 4: Neural Network Workflow ---")

    # Prepare scaled data (QuantileTransformer)
    # This might take a moment on the full dataset, but it caches.
    X_train_nn_full, y_train_nn_full, X_test_nn_full, _ = prepare_nn_data(
        load_cached_data=True
    )

    # Subsample same indices as before to maintain consistency for ensemble demo
    X_train_nn_demo = X_train_nn_full[indices]
    y_train_nn_demo = y_train_nn_full[indices]

    # Split for NN (same split indices as XGB for fair comparison/blending)
    X_tr_nn = X_train_nn_demo[:-val_size]
    y_tr_nn = y_train_nn_demo[:-val_size]
    X_val_nn = X_train_nn_demo[-val_size:]
    y_val_nn = y_train_nn_demo[-val_size:]

    # Create Datasets and Loaders
    train_dataset = ForestDataset(X_tr_nn, y_tr_nn)
    val_dataset = ForestDataset(X_val_nn, y_val_nn)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.NN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.NN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print("Training Neural Network on demo split...")
    input_dim = X_tr_nn.shape[1]
    nn_model = train_nn_fold(
        train_loader, val_loader, input_dim=input_dim, num_classes=Config.NUM_CLASSES
    )

    assert nn_model is not None, "Neural Network training failed"

    # Predict on Validation set
    nn_val_preds = predict_nn(nn_model, val_loader)
    print(f"NN Validation Predictions Shape: {nn_val_preds.shape}")
    assert nn_val_preds.shape == (
        val_size,
        Config.NUM_CLASSES,
    ), "NN prediction shape mismatch"

    # ---------------------------------------------------------
    # 5. Ensemble Optimization
    # ---------------------------------------------------------
    print("\n--- Step 5: Ensemble Optimization ---")

    # Dictionary of OOF (Validation) predictions
    oof_preds = {"xgboost": xgb_val_preds, "resnet": nn_val_preds}

    # Optimize weights
    # y_val_xgb and y_val_nn are identical (same indices)
    weights = optimize_blending_weights(oof_preds, y_val_xgb)

    # Verify weights
    total_weight = sum(weights.values())
    print(f"Sum of weights: {total_weight}")
    assert np.isclose(total_weight, 1.0), "Weights do not sum to 1.0"

    # Compute weighted average
    final_val_preds = weighted_average(
        [xgb_val_preds, nn_val_preds], [weights["xgboost"], weights["resnet"]]
    )

    print(f"Ensembled Prediction Shape: {final_val_preds.shape}")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n--- Step 6: Submission Generation (Example) ---")

    # Use the test subset predictions from XGB for demonstration
    # (In a real scenario, we would predict on full test set with both models and blend)

    # Convert probabilities to class labels
    # The models predict indices 0..5, we need to map back to original labels [1, 2, 3, 4, 6, 7]

    pred_indices = np.argmax(xgb_test_preds, axis=1)
    pred_labels = [Config.ORIGINAL_LABELS[i] for i in pred_indices]

    # Create DataFrame
    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids[:100], Config.TARGET_COL: pred_labels}
    )

    print("Sample Submission DataFrame:")
    print(submission_df.head())

    # Verify format
    assert submission_df.shape == (100, 2), "Submission shape incorrect"
    assert Config.ID_COL in submission_df.columns, "Id column missing"
    assert Config.TARGET_COL in submission_df.columns, "Target column missing"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
