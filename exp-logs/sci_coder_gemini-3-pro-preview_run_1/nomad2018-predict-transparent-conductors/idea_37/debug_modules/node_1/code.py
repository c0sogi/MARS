import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import prepare_data
from library.model import DualShellWideDeepSets
from library.engine import run_training, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_environment():
    """
    Creates a subset of the metadata to run a fast demo.
    Overrides Config parameters to use this subset and reduce runtime.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_meta_dir = "./working/demo_metadata"
    demo_work_dir = "./working/demo_execution"
    demo_sub_dir = "./working/demo_submission"

    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_work_dir, exist_ok=True)
    os.makedirs(demo_sub_dir, exist_ok=True)

    # Subset size
    N_SUBSET = 50

    # Read original metadata
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Create subsets if files exist (they should based on problem description)
    if os.path.exists(orig_train_path):
        df_train = pd.read_csv(orig_train_path)
        df_train.head(N_SUBSET).to_csv(
            os.path.join(demo_meta_dir, "train.csv"), index=False
        )
        print(
            f"Created subset train metadata with {len(df_train.head(N_SUBSET))} samples."
        )

    if os.path.exists(orig_val_path):
        df_val = pd.read_csv(orig_val_path)
        df_val.head(N_SUBSET).to_csv(
            os.path.join(demo_meta_dir, "val.csv"), index=False
        )
        print(f"Created subset val metadata with {len(df_val.head(N_SUBSET))} samples.")

    if os.path.exists(orig_test_path):
        df_test = pd.read_csv(orig_test_path)
        df_test.head(N_SUBSET).to_csv(
            os.path.join(demo_meta_dir, "test.csv"), index=False
        )
        print(
            f"Created subset test metadata with {len(df_test.head(N_SUBSET))} samples."
        )

    # Override Config parameters
    print("Overriding Config parameters for demo...")
    Config.METADATA_DIR = demo_meta_dir
    Config.WORKING_DIR = demo_work_dir
    Config.SUBMISSION_DIR = demo_sub_dir

    # Update paths based on new dirs
    Config.TRAIN_METADATA_PATH = os.path.join(demo_meta_dir, "train.csv")
    Config.VAL_METADATA_PATH = os.path.join(demo_meta_dir, "val.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_meta_dir, "test.csv")

    Config.TRAIN_DATA_PATH = os.path.join(demo_work_dir, "train_data.npz")
    Config.VAL_DATA_PATH = os.path.join(demo_work_dir, "val_data.npz")
    Config.TEST_DATA_PATH = os.path.join(demo_work_dir, "test_data.npz")
    Config.SCALERS_PATH = os.path.join(demo_work_dir, "scalers.npz")

    Config.MODEL_SAVE_PATH = os.path.join(demo_work_dir, "best_model.pt")
    Config.SUBMISSION_PATH = os.path.join(demo_sub_dir, "demo_submission.csv")

    # Reduce training intensity
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist via Config method (it uses the updated paths)
    Config.setup_directories()


def test_data_pipeline():
    """
    Tests data preparation and loading.
    """
    print("\n--- Testing Data Pipeline ---")
    # Force re-processing to ensure we use the subset
    train_loader, val_loader, test_loader = prepare_data(load_cached_data=False)

    # Verify we got loaders
    assert train_loader is not None, "Train loader is None"
    assert val_loader is not None, "Val loader is None"
    assert test_loader is not None, "Test loader is None"

    # Check a single batch
    batch = next(iter(train_loader))
    atom_feats, batch_indices, global_feats, targets, ids = batch

    print(f"Batch loaded successfully.")
    print(f"Atom features shape: {atom_feats.shape} (N_atoms, 9)")
    print(f"Global features shape: {global_feats.shape} (Batch, 12)")
    print(f"Targets shape: {targets.shape} (Batch, 2)")

    # Assertions
    assert (
        atom_feats.dim() == 2 and atom_feats.shape[1] == 9
    ), "Incorrect atom feature dim"
    assert (
        global_feats.dim() == 2 and global_feats.shape[1] == 12
    ), "Incorrect global feature dim"
    assert targets.dim() == 2 and targets.shape[1] == 2, "Incorrect target dim"
    assert (
        batch_indices.max() < global_feats.shape[0]
    ), "Batch indices exceed batch size"

    return train_loader, val_loader, test_loader


def test_model_architecture(train_loader):
    """
    Tests model instantiation and forward pass.
    """
    print("\n--- Testing Model Architecture ---")
    device = torch.device(Config.DEVICE)
    model = DualShellWideDeepSets().to(device)

    # Get a batch
    batch = next(iter(train_loader))
    atom_feats, batch_indices, global_feats, _, _ = batch

    # Move to device
    atom_feats = atom_feats.to(device)
    batch_indices = batch_indices.to(device)
    global_feats = global_feats.to(device)

    # Forward pass
    output = model(atom_feats, batch_indices, global_feats)

    print(f"Model output shape: {output.shape}")

    # Assertions
    assert output.shape == (global_feats.shape[0], 2), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    print("Model forward pass successful.")
    return model


def test_training_loop(train_loader, val_loader):
    """
    Tests the training engine.
    """
    print("\n--- Testing Training Loop ---")
    # This will train for Config.NUM_EPOCHS (set to 2 in setup)
    trained_model = run_training(train_loader, val_loader)

    # Check if model file was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("Training loop completed and model saved.")
    return trained_model


def test_inference(model, test_loader):
    """
    Tests submission generation.
    """
    print("\n--- Testing Inference and Submission ---")
    generate_submission(model, test_loader)

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify submission content
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df.shape}")
    print(f"Submission columns: {df.columns.tolist()}")

    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert list(df.columns) == expected_cols, "Submission columns mismatch"
    assert len(df) > 0, "Submission file is empty"

    print("Inference successful.")


if __name__ == "__main__":
    try:
        set_seed(42)

        # 1. Setup Environment
        setup_demo_environment()

        # 2. Data Pipeline
        train_loader, val_loader, test_loader = test_data_pipeline()

        # 3. Model Architecture
        # We perform a quick check, but run_training creates its own model instance
        test_model_architecture(train_loader)

        # 4. Training
        trained_model = test_training_loop(train_loader, val_loader)

        # 5. Inference
        test_inference(trained_model, test_loader)

        print("\nAll demonstration steps completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        # Raise to ensure the failure is registered
        raise e
