import os
import shutil
import warnings
import torch
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import RTDSModel
from library.train import run_training, generate_submission


def main():
    print("=== Starting Demonstration Script ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("[1] Setting up Configuration...")

    # Override Config paths to use a demo directory to avoid polluting real working dirs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Override Cache paths to be inside the demo working directory
    Config.TRAIN_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "cache", "train_data.npz"
    )
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "cache", "val_data.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "cache", "test_data.npz")

    # Override Model Save Path
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pt")

    # Override Submission Path
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for training/val to be fast

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    Config.setup()

    # Set random seed
    seed_everything(Config.SEED)
    print("Configuration configured and directories created.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # We use load_cached_data=False to force the processing logic to run
    # This will parse the geometry files and generate features
    print("Processing data and creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify Train Loader Batch
    try:
        batch = next(iter(train_loader))
        atom_x, glob_x, targets, batch_indices, ids = batch

        print(f"  Batch loaded successfully.")
        print(
            f"  atom_x shape: {atom_x.shape} (Expected: [N_atoms, {Config.ATOMIC_FEATURE_DIM}])"
        )
        print(
            f"  glob_x shape: {glob_x.shape} (Expected: [{Config.BATCH_SIZE}, {Config.GLOBAL_FEATURE_DIM}])"
        )
        print(
            f"  targets shape: {targets.shape} (Expected: [{Config.BATCH_SIZE}, {Config.NUM_TARGETS}])"
        )

        # Assertions
        assert atom_x.ndim == 2
        assert atom_x.shape[1] == Config.ATOMIC_FEATURE_DIM
        assert glob_x.ndim == 2
        assert glob_x.shape[1] == Config.GLOBAL_FEATURE_DIM
        assert targets.ndim == 2
        assert targets.shape[1] == Config.NUM_TARGETS
        assert len(ids) == Config.BATCH_SIZE

        print("  Data shapes are correct.")

    except Exception as e:
        print(f"  Data pipeline verification failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple check
    model = RTDSModel().to(device)

    # Run a forward pass with the batch fetched earlier
    try:
        outputs = model(atom_x.to(device), glob_x.to(device), batch_indices.to(device))
        print(f"  Forward pass successful.")
        print(
            f"  Output shape: {outputs.shape} (Expected: [{Config.BATCH_SIZE}, {Config.NUM_TARGETS}])"
        )

        assert outputs.shape == (Config.BATCH_SIZE, Config.NUM_TARGETS)
        assert not torch.isnan(outputs).any(), "Model output contains NaNs"

        print("  Model logic verified.")

    except Exception as e:
        print(f"  Model verification failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Demo)...")

    # run_training uses the Config settings we modified (2 epochs, 20 samples)
    # We use load_cached_data=True because we just populated the cache in step 2.
    try:
        run_training(load_cached_data=True, debug_size=Config.DEBUG_SAMPLE_SIZE)

        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"  Model saved successfully at: {Config.MODEL_SAVE_PATH}")
        else:
            raise FileNotFoundError("Model file was not created.")

    except Exception as e:
        print(f"  Training loop failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    # We force load_cached_data=False here because the previous test_loader (step 2)
    # was created with debug_size=20. For submission, we want the full test set.
    # The generate_submission function in library/train.py calls get_dataloaders with debug_size=None
    # for the test set, so it will attempt to process all test files.

    try:
        # We set load_cached_data=False to ensure it processes the full test set
        # (overwriting the partial cache from step 2)
        generate_submission(load_cached_data=False)

        if os.path.exists(Config.SUBMISSION_PATH):
            df = pd.read_csv(Config.SUBMISSION_PATH)
            print(f"  Submission file created at: {Config.SUBMISSION_PATH}")
            print(f"  Submission shape: {df.shape}")
            print("  Head:")
            print(df.head())

            # Basic validation
            assert df.shape[1] == 3
            assert "id" in df.columns
            assert "formation_energy_ev_natom" in df.columns
            assert "bandgap_energy_ev" in df.columns
            assert len(df) > 0
        else:
            raise FileNotFoundError("Submission file was not created.")

    except Exception as e:
        print(f"  Submission generation failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
