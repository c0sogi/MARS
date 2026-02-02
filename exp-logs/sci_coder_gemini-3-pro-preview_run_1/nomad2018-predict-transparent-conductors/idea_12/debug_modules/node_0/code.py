import os
import shutil
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import DataHandler
from library.model import PAWDS, collate_fn
from library.train import Trainer
from library.predict import generate_submission


def main():
    print("Starting PA-WDS Library Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring for fast demo execution...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_PATH_TRAIN = os.path.join(
        Config.WORKING_DIR, "cache", "train_data.npz"
    )
    Config.CACHE_PATH_VAL = os.path.join(Config.WORKING_DIR, "cache", "val_data.npz")
    Config.CACHE_PATH_TEST = os.path.join(Config.WORKING_DIR, "cache", "test_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pt")

    # Submission path
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size

    # Ensure directories exist
    os.makedirs(os.path.join(Config.WORKING_DIR, "cache"), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Initialize seed
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configured. Working dir: {Config.WORKING_DIR}")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Processing Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating DataHandler...")

    # Initialize DataHandler
    data_handler = DataHandler()

    # Force re-computation by ensuring cache doesn't exist or is overwritten
    # (The unique working dir handles isolation, but we can be explicit)
    if os.path.exists(Config.CACHE_PATH_TRAIN):
        print("Cache found (unexpected for new dir), proceeding...")

    # Get datasets
    # This will trigger processing of the first 50 samples of train/val/test
    train_dataset, val_dataset, test_dataset = data_handler.get_datasets()

    # Verification
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    assert (
        len(train_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit"
    assert (
        len(val_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Val dataset size exceeds debug limit"

    # Check a single item
    sample_item = train_dataset[0]
    print("Sample item keys:", sample_item.keys())
    assert "atomic_features" in sample_item
    assert "global_features" in sample_item
    assert "targets" in sample_item
    assert "id" in sample_item

    # Check shapes
    # Atomic features: (N_atoms, 9) -> OneHot(4) + Coords(3) + NN(1) + Pot(1)
    # Global features: (11,) -> LatLen(3) + Angles(3) + Vol(1) + Dens(1) + Comp(3)
    # Targets: (2,)
    print(f"Atomic features shape: {sample_item['atomic_features'].shape}")
    print(f"Global features shape: {sample_item['global_features'].shape}")
    print(f"Targets shape: {sample_item['targets'].shape}")

    assert sample_item["atomic_features"].shape[1] == Config.ATOMIC_INPUT_DIM
    assert sample_item["global_features"].shape[0] == Config.GLOBAL_INPUT_DIM
    assert sample_item["targets"].shape[0] == Config.OUTPUT_DIM

    # -------------------------------------------------------------------------
    # 3. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating PAWDS Model...")

    # Create a small dataloader for batch testing
    demo_loader = DataLoader(
        train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False
    )

    # Get a batch
    batch = next(iter(demo_loader))
    atomic_x = batch["atomic_features"].to(device)
    global_x = batch["global_features"].to(device)
    mask = batch["mask"].to(device)

    print(f"Batch atomic input shape: {atomic_x.shape}")  # (B, Max_N, 9)
    print(f"Batch global input shape: {global_x.shape}")  # (B, 11)
    print(f"Batch mask shape: {mask.shape}")  # (B, Max_N)

    # Instantiate model
    model = PAWDS().to(device)

    # Forward pass
    output = model(atomic_x, global_x, mask)
    print(f"Model output shape: {output.shape}")

    assert output.shape == (4, Config.OUTPUT_DIM), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # -------------------------------------------------------------------------
    # 4. Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Trainer...")

    # Re-create loaders with configured batch size
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Initialize Trainer
    trainer = Trainer(model, device)

    # Run training
    # Note: Config.NUM_EPOCHS is set to 2
    trainer.fit(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS)

    # Verify model file creation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 5. Prediction Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Prediction/Submission...")

    # Use generate_submission function
    # It internally loads test data and the saved model
    try:
        generate_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            debug_size=Config.DEBUG_SAMPLE_SIZE,
            device=device.type,
        )
    except Exception as e:
        print(f"Generate submission failed: {e}")
        raise e

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print("Columns:", df_sub.columns.tolist())

    # Check columns
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check row count (should match debug size or test set size)
    # Since we set DEBUG_SAMPLE_SIZE, it should match min(len(test_df), debug_size)
    # The test.csv has 240 samples. Debug size is 50.
    assert len(df_sub) == 50, f"Expected 50 predictions, got {len(df_sub)}"

    # Check values are valid (not NaN)
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
