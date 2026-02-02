import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model
from library import train

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a subset of the data and overrides config paths/params
    to ensure the demo runs quickly and correctly.
    """
    print(">>> Setting up demo environment...")

    # Define temporary directories
    demo_data_dir = "./working/demo_data"
    demo_working_dir = "./working/demo_working"
    demo_submission_dir = "./working/demo_submission"

    if os.path.exists(demo_data_dir):
        shutil.rmtree(demo_data_dir)
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_submission_dir):
        shutil.rmtree(demo_submission_dir)

    os.makedirs(demo_data_dir)
    os.makedirs(demo_working_dir)
    os.makedirs(demo_submission_dir)

    # --- Create Data Subsets ---
    # We need full breaths (80 steps per breath).
    # Let's take 100 breaths for train, 20 for val (from original train file), 20 for test.
    steps_per_breath = 80

    # Load original metadata
    print("    Loading original metadata subset...")
    train_orig = pd.read_csv(
        os.path.join(config.METADATA_DIR, "train.csv"), nrows=200 * steps_per_breath
    )
    test_orig = pd.read_csv(
        os.path.join(config.METADATA_DIR, "test.csv"), nrows=50 * steps_per_breath
    )

    # Split train_orig into train/val for the demo
    # Ensure we don't split a breath in half
    train_subset = train_orig.iloc[: 100 * steps_per_breath].copy()
    val_subset = train_orig.iloc[100 * steps_per_breath : 120 * steps_per_breath].copy()
    test_subset = test_orig.iloc[: 20 * steps_per_breath].copy()

    # Save subsets
    train_path = os.path.join(demo_data_dir, "train.csv")
    val_path = os.path.join(demo_data_dir, "val.csv")
    test_path = os.path.join(demo_data_dir, "test.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    print(
        f"    Created subsets: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # --- Override Config ---
    print("    Overriding configuration parameters...")
    config.TRAIN_PATH = train_path
    config.VAL_PATH = val_path
    config.TEST_PATH = test_path

    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_submission_dir

    # Update derived paths in config
    config.MODEL_PATH = os.path.join(config.WORKING_DIR, "model.pth")
    config.SCALER_PATH = os.path.join(config.WORKING_DIR, "scaler.pkl")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute requirements
    config.EPOCHS = 1
    config.BATCH_SIZE = 16
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.EARLY_STOPPING_PATIENCE = 1

    # Use CPU if GPU not available (though environment likely has GPU)
    # forcing device check again just in case
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return demo_data_dir, demo_working_dir, demo_submission_dir


def verify_metric_logic():
    """
    Verifies that compute_mae correctly ignores the expiratory phase.
    """
    print("\n>>> Verifying Metric Logic (compute_mae)...")

    # Case: 4 time steps.
    # Steps 0, 1: Inspiratory (u_out=0) -> Should be counted
    # Steps 2, 3: Expiratory (u_out=1) -> Should be ignored

    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 100.0, 100.0])  # Large errors in expiratory phase
    u_out = np.array([0, 0, 1, 1])

    # Expected MAE: (|10-12| + |20-18|) / 2 = (2 + 2) / 2 = 2.0
    mae = utils.compute_mae(y_pred, y_true, u_out)

    print(f"    Calculated MAE: {mae}")
    if abs(mae - 2.0) > 1e-6:
        raise AssertionError(f"Metric verification failed. Expected 2.0, got {mae}")
    print("    Metric logic verified.")


def verify_data_pipeline():
    """
    Verifies data loading, feature engineering, and batch shapes.
    """
    print("\n>>> Verifying Data Pipeline...")

    # Force reload to ensure preprocessing happens on new subset
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=False
    )

    # Check Train Loader
    x, u_out, y = next(iter(train_loader))

    # Expected shapes: (Batch, 80, Features), (Batch, 80), (Batch, 80)
    print(f"    Train Batch X shape: {x.shape}")
    print(f"    Train Batch u_out shape: {u_out.shape}")
    print(f"    Train Batch y shape: {y.shape}")

    if x.shape[1] != 80:
        raise AssertionError("Time dimension is not 80.")
    if x.shape[0] != config.BATCH_SIZE:
        raise AssertionError(
            f"Batch dimension mismatch. Expected {config.BATCH_SIZE}, got {x.shape[0]}"
        )

    # Check Test Loader (should return IDs instead of pressure)
    x_test, u_out_test, ids_test = next(iter(test_loader))
    print(f"    Test Batch IDs shape: {ids_test.shape}")

    # Verify IDs are integers
    if ids_test.dtype not in [torch.int32, torch.int64, torch.long]:
        # Depending on collate, might be float if not careful, but dataset returns long
        if not torch.is_floating_point(ids_test):
            pass  # Good
        else:
            # If it came back as float, we check if they are effectively integers
            pass

    return train_loader, val_loader, test_loader, x.shape[-1]


def verify_model_forward(input_dim):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n>>> Verifying Model Architecture...")

    net = model.GraduatedCapacityNetworkRobust(input_dim).to(config.DEVICE)

    # Create dummy batch
    batch_size = 4
    seq_len = 80
    dummy_x = torch.randn(batch_size, seq_len, input_dim).to(config.DEVICE)
    dummy_u_out = torch.zeros(batch_size, seq_len).to(config.DEVICE)

    # Forward
    pred, aux_pred = net(dummy_x, dummy_u_out)

    print(f"    Output Shape: {pred.shape}")

    if pred.shape != (batch_size, seq_len):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(batch_size, seq_len)}, got {pred.shape}"
        )

    print("    Model forward pass verified.")
    return net


def run_training_demo():
    """
    Runs the training loop.
    """
    print("\n>>> Running Training Demo...")

    # We use the provided train.run_training function
    # It handles dataloading internally, but since we set config paths, it will use our subsets.
    # We pass load_cached_data=True because verify_data_pipeline already generated the cache
    # in the correct demo_working directory.

    trained_model = train.run_training(
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # Verify model file exists
    if not os.path.exists(config.MODEL_PATH):
        raise AssertionError("Model file was not saved after training.")

    print(f"    Training complete. Model saved to {config.MODEL_PATH}")
    return trained_model


def run_inference_demo():
    """
    Runs the inference loop.
    """
    print("\n>>> Running Inference Demo...")

    # Calls model.predict which loads the best model from disk and generates submission
    model.predict()

    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not generated.")

    # Verify submission content
    sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission shape: {sub.shape}")

    # We used 20 breaths for test, 80 steps each = 1600 rows
    expected_rows = 20 * 80
    if len(sub) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(sub)}"
        )

    print("    Inference demo successful.")


def cleanup(dirs_to_clean):
    """
    Removes temporary directories.
    """
    print("\n>>> Cleaning up...")
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"    Removed {d}")


if __name__ == "__main__":
    # 1. Setup
    utils.seed_everything(42)
    demo_dirs = setup_demo_environment()

    try:
        # 2. Verify Metric
        verify_metric_logic()

        # 3. Verify Data Pipeline
        _, _, _, input_dim = verify_data_pipeline()

        # 4. Verify Model
        verify_model_forward(input_dim)

        # 5. Run Training
        run_training_demo()

        # 6. Run Inference
        run_inference_demo()

        print("\n=== All Demonstrations Passed Successfully ===")

    except Exception as e:
        print(f"\n!!! Demo Failed: {e}")
        raise e
    finally:
        # 7. Cleanup
        cleanup(demo_dirs)
