import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import prepare_datasets, VentilatorDataset
from library.model import WPABiLSTM
from library.loss import WeightedL1Loss
from library.train import Trainer
import library.inference as inference


def setup_demo_environment():
    """
    Creates a temporary directory and generates mini-datasets for rapid testing.
    """
    print("=== Setting up Demo Environment ===")

    # Define paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train_meta_path = os.path.join(demo_dir, "mini_train_meta.csv")
    mini_val_meta_path = os.path.join(demo_dir, "mini_val_meta.csv")
    mini_test_meta_path = os.path.join(demo_dir, "mini_test_meta.csv")

    # 1. Create Mini Train Data (20 breaths: 16 Train, 4 Val)
    # Each breath is approx 80 rows. Read enough rows to capture 20 breaths.
    print("Generating mini training data...")
    df_raw_train = pd.read_csv(Config.TRAIN_CSV, nrows=3000)
    unique_breaths = df_raw_train["breath_id"].unique()

    if len(unique_breaths) < 20:
        raise ValueError("Not enough breaths in the first 3000 rows of train.csv")

    train_bids = unique_breaths[:16]
    val_bids = unique_breaths[16:20]
    all_bids = np.concatenate([train_bids, val_bids])

    # Filter raw data
    df_mini_train = df_raw_train[df_raw_train["breath_id"].isin(all_bids)].copy()
    df_mini_train.to_csv(mini_train_path, index=False)

    # Create Metadata
    # Train Metadata
    df_meta_train = df_mini_train[df_mini_train["breath_id"].isin(train_bids)].copy()
    df_meta_train = df_meta_train[["id", "breath_id", "pressure"]]
    df_meta_train["source_file"] = (
        "mini_train.csv"  # Relative to input dir in real scenario, but we patch path
    )
    df_meta_train.to_csv(mini_train_meta_path, index=False)

    # Val Metadata
    df_meta_val = df_mini_train[df_mini_train["breath_id"].isin(val_bids)].copy()
    df_meta_val = df_meta_val[["id", "breath_id", "pressure"]]
    df_meta_val["source_file"] = "mini_train.csv"
    df_meta_val.to_csv(mini_val_meta_path, index=False)

    # 2. Create Mini Test Data (5 breaths)
    print("Generating mini test data...")
    df_raw_test = pd.read_csv(Config.TEST_CSV, nrows=1000)
    test_bids = df_raw_test["breath_id"].unique()[:5]

    df_mini_test = df_raw_test[df_raw_test["breath_id"].isin(test_bids)].copy()
    df_mini_test.to_csv(mini_test_path, index=False)

    # Test Metadata
    df_meta_test = df_mini_test[["id", "breath_id"]].copy()
    df_meta_test["source_file"] = "mini_test.csv"
    df_meta_test.to_csv(mini_test_meta_path, index=False)

    return (
        demo_dir,
        mini_train_path,
        mini_test_path,
        mini_train_meta_path,
        mini_val_meta_path,
        mini_test_meta_path,
    )


def patch_config(demo_dir, train_csv, test_csv, train_meta, val_meta, test_meta):
    """
    Modifies Config attributes to point to mini-datasets and optimize for speed.
    """
    print("=== Patching Configuration ===")

    # Paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CSV = train_csv
    Config.TEST_CSV = test_csv
    Config.TRAIN_META = train_meta
    Config.VAL_META = val_meta
    Config.TEST_META = test_meta
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")

    # Model Hyperparameters (Reduced for speed)
    Config.LSTM_HIDDEN_SIZE = 64
    Config.LSTM_LAYERS = 2
    Config.GLU_HIDDEN_SIZE = 32

    # Training Hyperparameters
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    print("Configuration patched successfully.")


def verify_dataset_pipeline():
    print("\n=== Verifying Dataset Pipeline ===")
    # prepare_datasets will now use the patched Config paths
    train_ds, val_ds, test_ds = prepare_datasets(load_cached_data=False)

    # Assertions
    # We selected 16 train breaths, 4 val breaths, 5 test breaths
    print(f"Train dataset size: {len(train_ds)} (Expected 16)")
    assert len(train_ds) == 16, f"Expected 16 train sequences, got {len(train_ds)}"

    print(f"Val dataset size: {len(val_ds)} (Expected 4)")
    assert len(val_ds) == 4, f"Expected 4 val sequences, got {len(val_ds)}"

    print(f"Test dataset size: {len(test_ds)} (Expected 5)")
    assert len(test_ds) == 5, f"Expected 5 test sequences, got {len(test_ds)}"

    # Check item shape: (Seq_Len, Features)
    # Seq_Len is 80. Features depends on Config (11 continuous + 1 binary = 12)
    sample_x, sample_y = train_ds[0]
    print(f"Sample X shape: {sample_x.shape}")
    print(f"Sample y shape: {sample_y.shape}")

    assert sample_x.shape[0] == 80, "Sequence length mismatch"
    assert sample_y.shape[0] == 80, "Target sequence length mismatch"
    assert (
        sample_x.shape[1] == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {sample_x.shape[1]}"

    return train_ds


def verify_model_logic():
    print("\n=== Verifying Model Logic ===")
    model = WPABiLSTM()
    model.eval()

    # Create dummy input: (Batch=2, Seq=80, Feat=INPUT_DIM)
    dummy_input = torch.randn(2, 80, Config.INPUT_DIM)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expected output: (Batch, Seq) -> (2, 80)
    assert output.shape == (
        2,
        80,
    ), f"Model output shape mismatch. Expected (2, 80), got {output.shape}"
    print("Model forward pass successful.")
    return model


def verify_loss_logic():
    print("\n=== Verifying Loss Logic ===")
    criterion = WeightedL1Loss()

    # Batch=1, Seq=5
    preds = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0]])
    targets = torch.tensor([[12.0, 12.0, 12.0, 12.0, 12.0]])

    # Create inputs with u_out.
    # u_out is the last feature in the concatenated vector (SegregatedScaler logic).
    # We need to find the index.
    u_out_idx = criterion.u_out_idx

    # Create dummy inputs
    inputs = torch.zeros(1, 5, Config.INPUT_DIM)

    # Set u_out: 0, 0, 1, 1, 1 (2 insp, 3 exp)
    inputs[0, 0, u_out_idx] = 0
    inputs[0, 1, u_out_idx] = 0
    inputs[0, 2, u_out_idx] = 1
    inputs[0, 3, u_out_idx] = 1
    inputs[0, 4, u_out_idx] = 1

    # Calculate expected loss manually
    # Error is |10-12| = 2.0 everywhere.
    # Weights: Insp=1.0, Exp=0.1
    # Weighted Errors:
    # idx 0: 2.0 * 1.0 = 2.0
    # idx 1: 2.0 * 1.0 = 2.0
    # idx 2: 2.0 * 0.1 = 0.2
    # idx 3: 2.0 * 0.1 = 0.2
    # idx 4: 2.0 * 0.1 = 0.2
    # Sum = 4.6, Mean = 4.6 / 5 = 0.92

    loss = criterion(preds, targets, inputs)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert (
        abs(loss.item() - 0.92) < 1e-5
    ), f"Loss calculation incorrect. Expected 0.92, got {loss.item()}"
    print("Loss logic verified.")


def run_full_training_cycle():
    print("\n=== Running Full Training Cycle (1 Epoch) ===")

    # Initialize Trainer
    # This will re-load datasets (using cached .npz files created in verify_dataset_pipeline)
    trainer = Trainer()

    # Run Fit
    trainer.fit()

    # Check if best model was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print("Training complete. Best model saved.")

    return trainer


def run_inference_verification():
    print("\n=== Running Inference Verification ===")

    # Run prediction
    inference.predict(batch_size=Config.BATCH_SIZE, num_workers=0)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # We have 5 test breaths * 80 steps = 400 rows
    expected_rows = 5 * 80
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission columns mismatch"

    print("Inference successful. Submission file verified.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup Environment
        paths = setup_demo_environment()

        # 2. Patch Config
        patch_config(*paths)

        # 3. Verify Dataset
        verify_dataset_pipeline()

        # 4. Verify Model
        verify_model_logic()

        # 5. Verify Loss
        verify_loss_logic()

        # 6. Run Training
        run_full_training_cycle()

        # 7. Run Inference
        run_inference_verification()

        print("\n=== All Demonstrations Passed Successfully ===")

    except Exception as e:
        print(f"\n!!! DEMONSTRATION FAILED: {e} !!!")
        raise e
