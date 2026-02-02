import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ISCI_CNN, load_and_process_data
from library.train_eval import train_model, make_submission
from library.utils import set_seed

if __name__ == "__main__":
    print("=== Starting Demo Script ===")

    # 1. Configuration Override for Speed
    # We set folds to 2 because train_model(debug=True) hardcodes n_folds=2.
    # We want make_submission to iterate exactly over the folds that were trained.
    print("Configuring parameters for rapid demonstration...")
    Config.NUM_FOLDS = 2
    Config.NUM_EPOCHS = 1  # Note: debug=True in train_model forces 2 epochs, but we set this for consistency
    set_seed(Config.SEED)

    # 2. Data Loading Verification
    print("\n[1/5] Verifying Data Loading and Processing...")
    # This will load from cache if available, or process raw json if not.
    X_train, y_train, ang_train, ids_train, X_test, ang_test, ids_test = (
        load_and_process_data()
    )

    # Assertions to ensure data integrity
    assert len(X_train) == len(y_train), "Mismatch between training images and labels"
    assert len(X_train) > 0, "Training data is empty"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected training data shape: {X_train.shape}"
    assert X_test.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected test data shape: {X_test.shape}"
    print(f"Data Loaded Successfully.")
    print(f"  Train Shape: {X_train.shape}")
    print(f"  Test Shape:  {X_test.shape}")

    # 3. Model Architecture Verification
    print("\n[2/5] Verifying Model Architecture...")
    device = Config.DEVICE
    model = ISCI_CNN().to(device)

    # Create dummy batch: 4 samples, 3 channels, 75x75
    dummy_images = torch.randn(4, 3, 75, 75).to(device)
    # Create dummy angles: 4 samples, 1 value each
    dummy_angles = torch.tensor([[35.0], [40.5], [30.2], [45.1]]).to(device)

    # Perform forward pass
    try:
        output = model(dummy_images, dummy_angles)
        # Output should be (Batch_Size, 1) logits
        assert output.shape == (
            4,
            1,
        ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
        print("Model forward pass successful.")
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}")

    # 4. Training Pipeline (Debug Mode)
    print("\n[3/5] Executing Training Pipeline (Debug Mode)...")
    # debug=True runs on a small subset (100 samples) for 2 epochs and 2 folds
    train_model(debug=True)

    # Verify checkpoints were created
    print("Verifying checkpoints...")
    for fold in range(Config.NUM_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")
    print("Training complete and checkpoints verified.")

    # 5. Inference and Submission
    print("\n[4/5] Generating Submission...")
    # This uses the trained checkpoints to predict on the full test set
    make_submission()

    # 6. Final Validation
    print("\n[5/5] Validating Submission File...")
    submission_path = Config.SUBMISSION_FILE

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check dimensions
    expected_rows = len(ids_test)
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission has {len(df_sub)} rows, expected {expected_rows}"
        )

    # Check columns
    required_cols = ["id", "is_iceberg"]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Check value range (probabilities)
    if not pd.api.types.is_numeric_dtype(df_sub["is_iceberg"]):
        raise TypeError("is_iceberg column is not numeric")

    print(f"Submission verified. File saved at: {submission_path}")
    print("\n=== Demo Completed Successfully ===")
