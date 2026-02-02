import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import set_seed, compute_metric
from library.data_factory import prepare_data, VentilatorDataset
from library.model_factory import WCMI_BiLSTM
from library.trainer import Trainer


def create_mini_dataset(num_breaths=50):
    """
    Creates a small subset of the training and test data for demonstration purposes.
    Updates Config paths to point to these new mini files.
    """
    print(f"Creating mini dataset with {num_breaths} breaths...")

    # Define paths for mini data
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_env")
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    # Each breath is 80 steps
    rows_to_read = num_breaths * 80

    # Read subset of raw data
    # We read slightly more to ensure we get complete breaths, then filter
    df_train = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "train.csv"), nrows=rows_to_read
    )
    df_test = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "test.csv"), nrows=rows_to_read
    )

    # Save mini raw files
    df_train.to_csv(mini_train_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    # Generate Mini Metadata
    # Split unique breaths 80/20
    unique_breaths = df_train["breath_id"].unique()
    split_idx = int(len(unique_breaths) * 0.8)
    train_breaths = unique_breaths[:split_idx]
    val_breaths = unique_breaths[split_idx:]

    # Create metadata dfs
    # Note: The original metadata generation script includes 'source_file',
    # but the library code reads raw CSVs based on Config paths,
    # and filters based on breath_id in metadata.

    train_meta = pd.DataFrame({"breath_id": train_breaths})
    val_meta = pd.DataFrame({"breath_id": val_breaths})

    mini_train_meta_path = os.path.join(demo_dir, "mini_train_meta.csv")
    mini_val_meta_path = os.path.join(demo_dir, "mini_val_meta.csv")

    train_meta.to_csv(mini_train_meta_path, index=False)
    val_meta.to_csv(mini_val_meta_path, index=False)

    # Update Config to use these mini files
    Config.TRAIN_CSV = mini_train_path
    Config.TEST_CSV = mini_test_path
    Config.TRAIN_META = mini_train_meta_path
    Config.VAL_META = mini_val_meta_path
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Small batch size for small data
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo


def verify_metric_logic():
    """
    Verifies that the compute_metric function correctly handles the inspiratory phase mask.
    """
    print("Verifying metric logic...")
    # Create synthetic data
    # 5 time steps
    preds = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])
    targets = torch.tensor([10.0, 12.0, 10.0, 15.0, 10.0])

    # u_out: 0 = Inspiratory (Scored), 1 = Expiratory (Ignored)
    # Mask: [0, 0, 1, 1, 0]
    # Indices scored: 0, 1, 4
    # Errors:
    # 0: |10-10| = 0
    # 1: |10-12| = 2
    # 2: Ignored (u_out=1)
    # 3: Ignored (u_out=1)
    # 4: |10-10| = 0
    # Mean Error: (0 + 2 + 0) / 3 = 0.666...
    u_out = torch.tensor([0, 0, 1, 1, 0])

    mae = compute_metric(preds, targets, u_out)
    expected_mae = 2.0 / 3.0

    assert np.isclose(
        mae, expected_mae
    ), f"Metric calculation failed. Expected {expected_mae}, got {mae}"
    print("Metric logic verified.")


def run_pipeline_demo():
    # 1. Setup
    set_seed(42)
    create_mini_dataset(num_breaths=100)

    # 2. Data Processing
    print("\n--- Running Data Processing ---")
    # Force processing from scratch by ensuring cache dir is clean or empty
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    (X_train, y_train, u_out_train), (X_val, y_val, u_out_val), (X_test, test_ids) = (
        prepare_data(load_cached_data=False)
    )

    # Assertions for Data
    print(f"Train shape: {X_train.shape}")
    assert len(X_train.shape) == 3, "X_train should be 3D (N, 80, Feats)"
    assert X_train.shape[1] == 80, "Sequence length must be 80"
    assert X_train.shape[0] > 0, "Train set is empty"
    assert y_train.shape == (X_train.shape[0], 80), "y_train shape mismatch"

    # Create Loaders
    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(X_test)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]

    model = WCMI_BiLSTM(input_dim).to(device)

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 80, input_dim).to(device)
    dummy_output = model(dummy_input)
    assert dummy_output.shape == (
        2,
        80,
    ), f"Model output shape mismatch. Expected (2, 80), got {dummy_output.shape}"
    print("Model architecture verified.")

    # 4. Training
    print("\n--- Starting Training Demo ---")
    trainer = Trainer(model, device)

    # Run fit (using reduced epochs from Config override)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not created."
    print("Training completed and model saved.")

    # 5. Inference
    print("\n--- Generating Submission ---")
    trainer.generate_submission(test_loader, test_ids)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape[0] == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {sub_df.shape[0]}"
    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission columns missing."
    print("Submission generated successfully.")


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Verify utility logic first
    verify_metric_logic()

    # Run the full pipeline
    run_pipeline_demo()

    print("\nAll demonstrations completed successfully.")
