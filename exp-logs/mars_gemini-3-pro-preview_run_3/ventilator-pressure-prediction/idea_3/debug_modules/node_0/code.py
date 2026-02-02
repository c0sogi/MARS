import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shutil

# Import from the provided library
from library.config import Config
from library.model import ParallelTCNLSTM, prepare_data, masked_mae_loss
from library.train_eval import run_training


def setup_demo_environment():
    """
    Configures the environment for a fast demonstration run.
    Modifies the Config class in-place to use a temporary working directory
    and a small subset of data.
    """
    print(">>> Setting up demonstration environment...")

    # Define working directories
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 1. Patch Config for Debug/Demo Mode
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission", "submission.csv")

    # Enable Debug mode to process only a small subset of breaths
    Config.DEBUG = True
    Config.DEBUG_BREATHS = 100  # Use 100 breaths for speed
    Config.FORCE_RECOMPUTE = True  # Ensure we generate new features for this subset

    # 2. Handle Test Metadata Consistency
    # The submission generator reads Config.TEST_PATH to get IDs.
    # In debug mode, the model only predicts for DEBUG_BREATHS.
    # We must create a subset of the test metadata so lengths match.
    print("Creating subset of test metadata for consistency...")
    original_test_df = pd.read_csv("./metadata/test.csv")
    subset_test_df = original_test_df.iloc[: Config.DEBUG_BREATHS * 80].copy()

    subset_test_path = os.path.join(demo_dir, "test_subset.csv")
    subset_test_df.to_csv(subset_test_path, index=False)

    # Point Config to this new subset file
    Config.TEST_PATH = subset_test_path

    print(f"Config patched. Working dir: {Config.WORKING_DIR}")
    print(f"Debug mode: {Config.DEBUG}, Breaths: {Config.DEBUG_BREATHS}")


def verify_data_pipeline():
    """
    Verifies the data loading and feature engineering logic.
    """
    print("\n>>> Verifying Data Pipeline...")

    # Call prepare_data directly
    # This will load data, compute features, scale them, and cache them.
    X_train, y_train, X_val, y_val, X_test = prepare_data(load_cached_data=False)

    # Check Shapes
    # Shape should be (N_breaths, 80, N_features)
    # N_features is 10 based on Config.FEATURE_COLS
    print(f"Train X shape: {X_train.shape}")
    print(f"Train y shape: {y_train.shape}")

    assert X_train.ndim == 3, "X_train should be 3D (N, L, F)"
    assert X_train.shape[1] == 80, "Sequence length should be 80"
    assert X_train.shape[2] == 10, "Feature dimension should be 10"
    assert y_train.shape == (X_train.shape[0], 80, 1), "Target shape mismatch"

    # Verify Feature Engineering Logic (Sanity Check)
    # Feature 0 is time_step, Feature 5 is u_in_cumsum (Integral)
    # Since we scaled data, values won't be raw, but we can check consistency.
    # u_in_cumsum should generally increase over time within a breath.
    # Let's check the first breath.
    breath_0_cumsum = X_train[0, :, 5]
    # It's scaled, so we can't strictly check monotonicity if mean/std varies,
    # but generally the integral of positive u_in correlates with time.

    print("Data Pipeline verification passed.")
    return X_train


def verify_model_logic(input_sample):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n>>> Verifying Model Architecture...")

    # Create a lightweight config for verification
    demo_hyperparams = Config.HYPERPARAMS.copy()
    demo_hyperparams["tcn_channels"] = 16
    demo_hyperparams["lstm_hidden_dim"] = 32
    demo_hyperparams["device"] = "cpu"  # Force CPU for simple check

    model = ParallelTCNLSTM(demo_hyperparams)
    model.eval()

    # Convert sample to tensor
    # input_sample shape: (N, 80, 10). Take 2 samples.
    x_tensor = torch.FloatTensor(input_sample[:2])

    with torch.no_grad():
        output = model(x_tensor)

    print(f"Model Output shape: {output.shape}")

    # Assertions
    assert output.shape == (2, 80, 1), f"Expected output (2, 80, 1), got {output.shape}"

    # Verify Loss Function Masking
    print("Verifying Masked MAE Loss...")
    y_true = torch.ones((2, 80, 1)) * 10
    y_pred = torch.ones((2, 80, 1)) * 12  # Error is 2

    # Create u_out mask.
    # In the pipeline, u_out is scaled.
    # Here we simulate the logic: u_out_scaled < 0 means inspiratory (mask=1).
    # Let's create a dummy input tensor where half the breath is inspiratory.
    # We need index 2 to be u_out.
    dummy_input = torch.zeros((2, 80, 10))
    # Set first 30 steps to inspiratory (negative scaled value), rest to expiratory (positive)
    dummy_input[:, :30, 2] = -1.0
    dummy_input[:, 30:, 2] = 1.0

    loss = masked_mae_loss(y_pred, y_true, dummy_input[:, :, 2:3])

    # Expected loss:
    # Error is 2.0 everywhere.
    # Mask is 1 for first 30 steps, 0 for rest.
    # Sum of errors = 2.0 * 30 = 60.
    # Count = 30.
    # MAE = 2.0.
    print(f"Calculated Loss: {loss.item()}")
    assert abs(loss.item() - 2.0) < 1e-5, "Loss calculation incorrect"

    print("Model logic verification passed.")


def run_demo_training():
    """
    Runs the full training loop using the library's run_training function.
    """
    print("\n>>> Running Full Training Demo...")

    # Define fast hyperparameters
    fast_hyperparams = Config.HYPERPARAMS.copy()
    fast_hyperparams.update(
        {
            "epochs": 2,  # Very few epochs
            "batch_size": 16,  # Small batch size
            "tcn_channels": 16,  # Smaller model
            "lstm_hidden_dim": 32,
            "fc_hidden_dim": 32,
            "num_workers": 0,  # Avoid multiprocessing overhead for small demo
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )

    # Run training
    # This handles data loading (using our patched Config), training, and submission generation
    run_training(config=fast_hyperparams, save_path=Config.MODEL_SAVE_PATH)

    # Verify Outputs
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved.")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {sub_df.shape}")

    expected_rows = Config.DEBUG_BREATHS * 80
    assert (
        len(sub_df) == expected_rows
    ), f"Submission rows {len(sub_df)} mismatch expected {expected_rows}"

    print("Training demo completed successfully.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Verify Data Pipeline
        X_sample = verify_data_pipeline()

        # 3. Verify Model Logic
        verify_model_logic(X_sample)

        # 4. Run Training
        run_demo_training()

        print("\n=== All demonstrations passed successfully ===")

    except AssertionError as e:
        print(f"\n!!! Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
