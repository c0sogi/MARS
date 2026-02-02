import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shutil
import sys

# Import provided library modules
import library.config as config
import library.utils as utils
import library.features as features
import library.data_loader as data_loader
import library.model as model_lib
import library.engine as engine


def create_subset(source_path, dest_path, num_breaths):
    """
    Reads the first N breaths from the source CSV and saves them to dest_path.
    Assumes 80 time steps per breath.
    """
    rows_to_read = num_breaths * 80
    # Read header first to ensure we have columns
    df = pd.read_csv(source_path, nrows=rows_to_read)

    # Verify we didn't split a breath (though nrows logic should be safe given data structure)
    assert len(df) % 80 == 0, f"Subset length {len(df)} is not divisible by 80"

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df.to_csv(dest_path, index=False)
    print(f"Created subset: {dest_path} with {len(df)} rows ({num_breaths} breaths)")
    return len(df)


def verify_loss_function():
    print("\n=== Verifying MaskedMAELoss ===")
    loss_fn = utils.MaskedMAELoss()

    # Create dummy data: Batch=2, Time=5
    # Breath 1: All inspiratory (u_out=0), Error=1.0
    # Breath 2: All expiratory (u_out=1), Error=100.0 (Should be ignored)
    y_pred = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0], [100.0, 100.0, 100.0, 100.0, 100.0]]
    )
    y_true = torch.zeros_like(y_pred)
    u_out = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])

    loss = loss_fn(y_pred, y_true, u_out)

    # Expected:
    # Breath 1 contributes |1-0| = 1 per step. Sum = 5. Valid steps = 5.
    # Breath 2 contributes 0 (masked). Valid steps = 0.
    # Total Sum = 5. Total Valid Steps = 5. Loss = 1.0.

    print(f"Calculated Loss: {loss.item()}")
    assert abs(loss.item() - 1.0) < 1e-6, "Loss calculation incorrect for masked data"
    print("MaskedMAELoss verification passed.")


def main():
    # 1. Setup Directories
    print("=== Setting up Demo Environment ===")
    demo_input_dir = "./working/demo_input"
    demo_working_dir = "./working/demo_run"

    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # 2. Create Data Subsets
    # We use the metadata files which are guaranteed to exist
    train_subset_path = os.path.join(demo_input_dir, "train.csv")
    val_subset_path = os.path.join(demo_input_dir, "val.csv")
    test_subset_path = os.path.join(demo_input_dir, "test.csv")

    # Create subsets (50 breaths train, 20 val, 20 test)
    create_subset("./metadata/train.csv", train_subset_path, 50)
    create_subset("./metadata/validation.csv", val_subset_path, 20)
    create_subset("./metadata/test.csv", test_subset_path, 20)

    # 3. Monkey Patch Library Configuration
    # We modify the module attributes to point to our subsets and working dir
    print("\n=== Patching Library Configuration ===")

    # Patch data_loader module
    data_loader.TRAIN_PATH = train_subset_path
    data_loader.VAL_PATH = val_subset_path
    data_loader.TEST_PATH = test_subset_path
    data_loader.WORKING_DIR = demo_working_dir
    data_loader.BATCH_SIZE = 16  # Small batch size for demo

    # Patch features module
    features.CACHE_DIR = demo_working_dir

    # 4. Verify Feature Engineering
    print("\n=== Verifying Feature Engineering ===")
    # We explicitly call engineer_features on the test subset to check logic
    df_test_feats = features.engineer_features(
        test_subset_path, "test_features_debug", load_cached_data=False
    )

    # Check dimensions
    expected_cols = config.MODEL_FEATURES
    assert all(
        col in df_test_feats.columns for col in expected_cols
    ), "Missing model features"
    assert "u_in_diff1" in df_test_feats.columns, "Derivative feature missing"
    assert "area" in df_test_feats.columns, "Physical feature missing"
    print(f"Feature Engineering successful. Shape: {df_test_feats.shape}")

    # 5. Data Loading
    print("\n=== Initializing Data Loaders ===")
    # Force processing from scratch with load_cached_data=False
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        load_cached_data=False
    )

    # Verify Loader Output
    sample_batch = next(iter(train_loader))
    x_batch, y_batch, u_out_batch = sample_batch

    print(f"Batch X shape: {x_batch.shape}")  # (Batch, 80, Features)
    print(f"Batch Y shape: {y_batch.shape}")  # (Batch, 80)

    assert x_batch.shape[1] == 80, "Sequence length mismatch"
    assert x_batch.shape[2] == config.INPUT_DIM, "Feature dimension mismatch"
    assert y_batch.shape == u_out_batch.shape, "Target and mask shape mismatch"

    # 6. Model Initialization
    print("\n=== Initializing Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = model_lib.KARHNet().to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = x_batch.to(device)
        dummy_out = model(dummy_input)
        print(f"Model Output Shape: {dummy_out.shape}")
        assert dummy_out.shape == (
            data_loader.BATCH_SIZE,
            80,
        ), "Model output shape incorrect"

    # 7. Loss Verification
    verify_loss_function()

    # 8. Training Loop Demonstration
    print("\n=== Starting Training Loop (1 Epoch) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = utils.MaskedMAELoss()

    train_loss = engine.train_fn(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        max_grad_norm=1.0,
    )
    print(f"Training Loss: {train_loss:.4f}")

    # 9. Evaluation
    print("\n=== Starting Evaluation ===")
    val_loss = engine.eval_fn(
        model=model, data_loader=val_loader, device=device, loss_fn=loss_fn
    )
    print(f"Validation Loss: {val_loss:.4f}")

    # 10. Prediction
    print("\n=== Generating Predictions ===")
    preds = engine.predict_fn(model, test_loader, device)

    # Verify predictions match input size
    # We used 20 breaths for test, 80 steps each = 1600 predictions
    expected_preds = 20 * 80
    print(f"Predictions generated: {len(preds)}")
    assert (
        len(preds) == expected_preds
    ), f"Expected {expected_preds} predictions, got {len(preds)}"

    # Create submission dataframe
    # We need the IDs. In a real scenario, we'd load them from the test file or cache.
    # The data_loader saves test_ids.npy to cache.
    test_ids_path = os.path.join(demo_working_dir, "test_ids.npy")
    test_ids = np.load(test_ids_path).flatten()

    submission = pd.DataFrame({"id": test_ids, "pressure": preds})

    submission_path = os.path.join(demo_working_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
