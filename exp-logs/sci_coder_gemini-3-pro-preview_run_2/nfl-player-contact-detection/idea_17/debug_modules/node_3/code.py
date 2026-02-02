import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.features import FeatureEngine
from library.model import MMWIN, train_model, optimize_threshold, predict
from library.trainer import ContactDataset


def main():
    # =========================================================================
    # 1. Configuration Overrides for Fast Execution
    # =========================================================================
    print("1. Configuring environment for fast demo execution...")

    # Override Config defaults to run on a small subset of data
    Config.USE_ALL_DATA = False
    Config.DEBUG_SAMPLE_SIZE = 500  # Process only ~500 rows/groups for speed

    # Reduce training complexity
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.HIDDEN_DIM = 64  # Smaller model for demo
    Config.NUM_LAYERS = 2

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Feature Engineering
    # =========================================================================
    print("\n2. Demonstrating FeatureEngine...")
    engine = FeatureEngine()

    # generate_features orchestrates loading, preprocessing, alignment, and windowing.
    # We set load_cached_data=False to force the pipeline to run from scratch
    # to demonstrate the logic works.
    train_df, val_df, test_df = engine.generate_features(load_cached_data=False)

    print(f"   Train DataFrame shape: {train_df.shape}")
    print(f"   Val DataFrame shape:   {val_df.shape}")
    print(f"   Test DataFrame shape:  {test_df.shape}")

    # Validation: Check for critical columns and data integrity
    assert not train_df.empty, "Training dataframe is empty."
    assert "contact" in train_df.columns, "Target column 'contact' missing."
    assert (
        "visual_iou_t0" in train_df.columns
    ), "Interaction feature 'visual_iou' missing."
    # Check if windowing created time-shifted columns (e.g., _t-1, _t+1)
    assert any(
        "_t-1" in col for col in train_df.columns
    ), "Time-shifted features missing."

    # =========================================================================
    # 3. Data Preparation
    # =========================================================================
    print("\n3. Preparing Data for Model...")

    # Define metadata columns to exclude from features
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
    ]

    # Select feature columns
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    print(f"   Selected {len(feature_cols)} features.")

    # Convert to Numpy
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["contact"].values.astype(np.float32)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["contact"].values.astype(np.float32)

    # Scale features (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create PyTorch Datasets
    train_dataset = ContactDataset(X_train, y_train)
    val_dataset = ContactDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # =========================================================================
    # 4. Model Initialization
    # =========================================================================
    print("\n4. Initializing MMWIN Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    model = MMWIN(
        input_dim=X_train.shape[1],
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    )
    model.to(device)

    # Validation: Verify forward pass dimensions
    dummy_input = torch.randn(4, X_train.shape[1]).to(device)
    dummy_output = model(dummy_input)
    assert dummy_output.shape == (4, 1), f"Output shape mismatch: {dummy_output.shape}"
    print("   Forward pass verification successful.")

    # =========================================================================
    # 5. Training
    # =========================================================================
    print("\n5. Training Model...")
    # train_model handles the training loop, focal loss, and early stopping
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
    )

    # =========================================================================
    # 6. Threshold Optimization
    # =========================================================================
    print("\n6. Optimizing Threshold...")
    # Finds the threshold that maximizes MCC on validation set
    best_threshold = optimize_threshold(model, val_loader, device)

    # Validation: Threshold should be a valid probability
    assert 0.0 < best_threshold < 1.0, f"Invalid threshold: {best_threshold}"

    # =========================================================================
    # 7. Inference on Test Set
    # =========================================================================
    print("\n7. Running Inference on Test Set...")

    # Prepare Test Data
    X_test = test_df[feature_cols].values.astype(np.float32)
    # Important: Use the SAME scaler fitted on train
    X_test = scaler.transform(X_test)

    test_dataset = ContactDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Generate predictions
    predictions = predict(model, test_loader, device, threshold=best_threshold)

    # Validation: Check prediction shape
    assert len(predictions) == len(test_df), "Prediction count mismatch."
    assert set(np.unique(predictions)).issubset({0, 1}), "Predictions must be binary."

    print(f"   Generated {len(predictions)} predictions.")
    print(
        f"   Positive predictions: {predictions.sum()} ({predictions.mean():.4f} rate)"
    )

    # =========================================================================
    # 8. Metric Verification
    # =========================================================================
    print("\n8. Verifying Metric Calculation...")
    # Verify compute_mcc with dummy data
    y_true_dummy = np.array([1, 0, 1, 1, 0, 0])
    y_pred_dummy = np.array([1, 0, 0, 1, 0, 1])
    mcc_score = compute_mcc(y_true_dummy, y_pred_dummy)

    print(f"   Dummy MCC Score: {mcc_score:.4f}")
    assert -1.0 <= mcc_score <= 1.0, "MCC score out of valid range [-1, 1]."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
