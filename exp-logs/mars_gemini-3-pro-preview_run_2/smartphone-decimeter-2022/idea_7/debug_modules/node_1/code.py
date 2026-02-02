import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library modules
# Note: We assume the library files are in a package named 'library' in the current directory.
from library.config import Config
from library.preprocessing import process_dataset
from library.dataset import get_dataset
from library.model import LocalAttentionTransformer
from library.engine import train_loop, generate_submission


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing demonstration script...")
    set_seed(42)

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed/Demo
    # ---------------------------------------------------------
    print("Configuring environment for fast execution...")

    # Create a separate working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR

    # Update file paths to point to the demo directory
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_data.parquet")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_data.parquet")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_data.parquet")
    Config.SCALER_PATH = os.path.join(DEMO_DIR, "scaler.json")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "lat_model.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2  # Use only 2 trips per split
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_ENCODER_LAYERS = 1  # Reduce model complexity for demo
    Config.NHEAD = 2

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Debug mode: {Config.DEBUG}")

    # ---------------------------------------------------------
    # 2. Data Processing
    # ---------------------------------------------------------
    print("\n--- Step 2: Data Processing ---")

    # Process Train Data
    # This loads metadata, samples trips (due to DEBUG=True), loads GNSS, aligns, and saves parquet
    print("Processing Training Data...")
    train_df = process_dataset("train", load_cached_data=False)
    assert os.path.exists(Config.TRAIN_CACHE_PATH), "Train cache file not created."
    assert not train_df.empty, "Processed training dataframe is empty."
    print(f"Train DataFrame shape: {train_df.shape}")

    # Process Validation Data
    print("Processing Validation Data...")
    val_df = process_dataset("val", load_cached_data=False)
    assert os.path.exists(Config.VAL_CACHE_PATH), "Val cache file not created."

    # Process Test Data
    print("Processing Test Data...")
    test_df = process_dataset("test", load_cached_data=False)
    assert os.path.exists(Config.TEST_CACHE_PATH), "Test cache file not created."

    # ---------------------------------------------------------
    # 3. Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n--- Step 3: Dataset & DataLoader ---")

    # Create Datasets
    # get_dataset handles scaler fitting/loading internally
    train_dataset = get_dataset("train", load_cached_data=True)
    val_dataset = get_dataset("val", load_cached_data=True)
    test_dataset = get_dataset("test", load_cached_data=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify a single batch
    sample_x, sample_y, sample_meta = next(iter(train_loader))
    print(
        f"Sample batch X shape: {sample_x.shape}"
    )  # Should be (Batch, Window, Features)
    print(f"Sample batch Y shape: {sample_y.shape}")  # Should be (Batch, 2)

    expected_x_shape = (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
        len(Config.INPUT_FEATURES),
    )
    assert (
        sample_x.shape == expected_x_shape
    ), f"Expected X shape {expected_x_shape}, got {sample_x.shape}"
    assert sample_y.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Expected Y shape {(Config.BATCH_SIZE, 2)}, got {sample_y.shape}"

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n--- Step 4: Model Initialization ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LocalAttentionTransformer().to(device)

    # Verify forward pass
    dummy_input = torch.randn(
        Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM
    ).to(device)
    dummy_output = model(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")
    assert dummy_output.shape == (Config.BATCH_SIZE, 2), "Model output shape mismatch."

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n--- Step 5: Training Loop ---")

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # We use the library's train_loop function which handles the loop, validation, and saving
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print("Training completed and model saved.")

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n--- Step 6: Inference & Submission ---")

    # Generate submission using the trained model
    generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_FILE,
    )

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission DataFrame shape: {sub_df.shape}")
    print("Submission head:")
    print(sub_df.head())

    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
