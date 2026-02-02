import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import Subset, DataLoader

# Ensure the current directory is in the python path to import library modules
sys.path.append(".")

# Import classes and functions from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders
from library.model_utils import ParallelDCNResNet
from library.train_utils import train_model, predict_and_submit

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("----------------------------------------------------------------")
    print("Starting Demonstration of Forest Cover Type Prediction Pipeline")
    print("----------------------------------------------------------------")

    # 1. Configure for Demo (Speed Optimization)
    # We override Config attributes to use a separate working directory and fast hyperparameters
    print("\n[Step 1] Configuring parameters for fast demonstration...")

    Config.WORKING_DIR = "./working/demo_execution/cache"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute load for demonstration
    Config.BATCH_SIZE = 128
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce overhead for small data

    print(f"  Working Directory: {Config.WORKING_DIR}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")

    # 2. Data Loading
    # We call get_dataloaders to trigger the full processing pipeline.
    # load_cached_data=False ensures we demonstrate the processing logic (loading parquet, feature engineering).
    print("\n[Step 2] Loading and Processing Data...")
    train_loader_full, val_loader_full, test_loader_full, input_dim, test_ids_full = (
        get_dataloaders(load_cached_data=False)
    )

    print(f"  Full Training Set Size: {len(train_loader_full.dataset)}")
    print(f"  Input Dimension: {input_dim}")

    # 3. Create Subsets for Speed
    # To ensure the script finishes quickly, we create subsets of the data (500 samples each).
    print("\n[Step 3] Creating Mini-Batches for Rapid Execution...")
    subset_size = 500

    # Create subsets using indices
    train_subset = Subset(train_loader_full.dataset, range(subset_size))
    val_subset = Subset(val_loader_full.dataset, range(subset_size))
    test_subset = Subset(test_loader_full.dataset, range(subset_size))

    # Slice test_ids to match the test subset length (required for submission generation)
    test_ids_subset = test_ids_full[:subset_size]

    # Create new DataLoaders for the subsets
    mini_train_loader = DataLoader(
        train_subset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    mini_val_loader = DataLoader(
        val_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    mini_test_loader = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    print(f"  Mini Train Size: {len(train_subset)}")
    print(f"  Mini Val Size: {len(val_subset)}")
    print(f"  Mini Test Size: {len(test_subset)}")

    # 4. Model Instantiation and Verification
    # We manually instantiate the model to verify the architecture and forward pass.
    print("\n[Step 4] Instantiating and Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)

    # Instantiate model with demo parameters
    model = ParallelDCNResNet(
        input_dim=input_dim, num_classes=7, hidden_dim=128, num_blocks=2, dropout=0.1
    ).to(device)

    # Create dummy input to check forward pass
    dummy_input = torch.randn(10, input_dim).to(device)
    dummy_output = model(dummy_input)

    # Assert output shape is (Batch Size, Num Classes)
    assert dummy_output.shape == (
        10,
        7,
    ), f"Model output shape mismatch. Expected (10, 7), got {dummy_output.shape}"
    print("  Model architecture verification passed (Output Shape: (10, 7)).")

    # 5. Training Loop
    # Execute the training utility. This function handles optimizer setup, loop, and checkpointing.
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    # train_model will instantiate a new model internally using Config parameters.
    # Since we updated Config earlier, it will use 1 Epoch.
    trained_model = train_model(
        mini_train_loader,
        mini_val_loader,
        input_dim=input_dim,
        num_classes=7,
        epochs=Config.EPOCHS,
    )

    print("  Training complete.")

    # 6. Prediction and Submission
    # Use the trained model to generate predictions on the test subset.
    print("\n[Step 6] Generating Predictions and Submission File...")
    predict_and_submit(trained_model, mini_test_loader, test_ids_subset)

    # 7. Final Verification
    # Verify that the submission file exists and has the correct format.
    print("\n[Step 7] Verifying Submission Artifacts...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {df_sub.shape}")

    # Validations
    assert df_sub.shape == (
        subset_size,
        2,
    ), f"Submission shape mismatch. Expected ({subset_size}, 2), got {df_sub.shape}"

    assert list(df_sub.columns) == [
        "Id",
        "Cover_Type",
    ], f"Submission columns mismatch. Got {list(df_sub.columns)}"

    assert df_sub["Id"].nunique() == subset_size, "Duplicate IDs found in submission."

    assert not df_sub.isnull().values.any(), "NaN values found in submission."

    print("  Verification successful.")
    print("\n----------------------------------------------------------------")
    print("Demonstration Completed Successfully.")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    # Set fixed seeds for reproducibility (already handled in libs, but good practice)
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
