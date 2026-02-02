import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed, get_logger
from library.data_loader import get_dataloaders
from library.model import SIRDS_SP
from library.trainer import Trainer


def main():
    # 1. Setup
    print("Setting up demonstration...")
    set_seed(42)

    # Define demo directories within the working directory
    DEMO_DIR = "./working/demo_execution"
    DEMO_METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Clean up previous runs if any to ensure a fresh start
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # 2. Create a small subset of data for speed optimization
    print("Creating data subsets...")

    # Read original metadata
    # Assuming the metadata files exist as per the problem description
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Original metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    orig_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(Config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample small subsets (e.g., 20 samples for train, 10 for val/test)
    # This ensures the demo runs very quickly
    demo_train = orig_train.head(20)
    demo_val = orig_val.head(10)
    demo_test = orig_test.head(10)

    # Save subsets to the demo metadata directory
    demo_train_path = os.path.join(DEMO_METADATA_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_METADATA_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_METADATA_DIR, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(
        f"Created subsets: Train={len(demo_train)}, Val={len(demo_val)}, Test={len(demo_test)}"
    )

    # 3. Override Config for Demo
    # We modify the Config class attributes directly to point to our demo data and settings
    print("Overriding Config parameters...")
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    Config.TRAIN_DATA_CACHE = os.path.join(DEMO_CACHE_DIR, "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(DEMO_CACHE_DIR, "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(DEMO_CACHE_DIR, "test_data.npz")

    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "demo_model.pt")
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "demo_submission.csv")

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 1

    # 4. Test Data Loading
    print("Testing Data Loading...")
    # load_cached_data=False forces processing the new small subset files instead of loading old cache
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead in this script
    )

    # Verify batch structure
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())

    # Assertions to verify data loading logic
    assert "atomic_x" in batch, "Batch missing atomic_x"
    assert "global_x" in batch, "Batch missing global_x"
    assert "symmetry_x" in batch, "Batch missing symmetry_x"
    assert "y" in batch, "Batch missing targets y"
    assert "atomic_mask" in batch, "Batch missing atomic_mask"

    # Check shapes
    # atomic_x: (Batch, Max_Atoms, 8)
    assert batch["atomic_x"].ndim == 3 and batch["atomic_x"].shape[2] == 8
    # global_x: (Batch, 11)
    assert batch["global_x"].ndim == 2 and batch["global_x"].shape[1] == 11
    # y: (Batch, 2)
    assert batch["y"].ndim == 2 and batch["y"].shape[1] == 2

    print(f"Atomic input shape: {batch['atomic_x'].shape}")
    print(f"Global input shape: {batch['global_x'].shape}")
    print(f"Target shape: {batch['y'].shape}")

    # 5. Test Model Instantiation and Forward Pass
    print("Testing Model...")
    model = SIRDS_SP()
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Move batch to device for testing forward pass
    atomic_x = batch["atomic_x"].to(device)
    atomic_mask = batch["atomic_mask"].to(device)
    global_x = batch["global_x"].to(device)
    symmetry_x = batch["symmetry_x"].to(device)

    # Forward pass
    output = model(atomic_x, atomic_mask, global_x, symmetry_x)
    print(f"Model output shape: {output.shape}")

    assert output.shape == (
        batch["atomic_x"].shape[0],
        Config.NUM_TARGETS,
    ), f"Expected output shape {(batch['atomic_x'].shape[0], Config.NUM_TARGETS)}, got {output.shape}"

    # 6. Test Trainer (Training Loop)
    print("Testing Trainer (Fit)...")
    trainer = Trainer(model, device=device)

    # Run training for limited epochs
    trainer.fit(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS)

    # Check if model checkpoint was created
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(f"Model checkpoint saved at {Config.MODEL_CHECKPOINT_PATH}")
    else:
        # Force save if not saved automatically (e.g. if loss didn't decrease, though unlikely with random init)
        print(
            "Checkpoint not found (likely due to short run). Saving manually for prediction test."
        )
        torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)

    # 7. Test Prediction
    print("Testing Prediction...")
    trainer.predict(test_loader, output_path=Config.SUBMISSION_PATH)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        preds_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Prediction rows: {len(preds_df)}")

        # Verify output format
        assert "id" in preds_df.columns
        assert "formation_energy_ev_natom" in preds_df.columns
        assert "bandgap_energy_ev" in preds_df.columns
        assert len(preds_df) == len(
            demo_test
        ), f"Expected {len(demo_test)} predictions, got {len(preds_df)}"
        print("Submission format verified.")
    else:
        raise AssertionError("Submission file was not created.")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
