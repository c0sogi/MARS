import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, rmsle
from library.features import process_dataset, get_scalers, scale_features
from library.data import get_train_val_loaders, get_test_loader
from library.model import CRRD_DeepSets
from library.engine import fit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a temporary environment with a subset of data for demonstration.
    Overrides Config paths to use this temporary environment.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    cache_dir = "./working/demo_cache"
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = cache_dir
    Config.CACHE_SCALERS = "scalers.npz"
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "demo_model.pt")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.HIDDEN_DIM_ATOMIC = 32
    Config.HIDDEN_DIM_GLOBAL = 16
    Config.HIDDEN_DIM_FUSION = 16

    # Create mini metadata files
    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample a small subset (e.g., 50 samples for train, 10 for val/test)
    mini_train = orig_train.head(50)
    mini_val = orig_val.head(10)
    mini_test = orig_test.head(10)

    # Save mini metadata
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update Config to point to mini metadata
    Config.METADATA_TRAIN_PATH = mini_train_path
    Config.METADATA_VAL_PATH = mini_val_path
    Config.METADATA_TEST_PATH = mini_test_path

    print(f"Demo environment configured in {demo_dir}")
    print(
        f"Train samples: {len(mini_train)}, Val samples: {len(mini_val)}, Test samples: {len(mini_test)}"
    )


def test_data_processing():
    """
    Demonstrates and validates the data processing pipeline.
    """
    print("\n--- Testing Data Processing ---")

    # Process training data (this will cache it)
    # We force load_cached_data=False to ensure processing logic runs
    data_dict = process_dataset(
        Config.METADATA_TRAIN_PATH, load_cached_data=False, mode="train"
    )

    # Validate dictionary structure
    expected_keys = [
        "atomic_inputs",
        "global_inputs",
        "batch_indices",
        "targets",
        "ids",
    ]
    for key in expected_keys:
        assert key in data_dict, f"Missing key {key} in processed data"

    # Validate shapes
    n_atoms = data_dict["atomic_inputs"].shape[0]
    n_samples = data_dict["global_inputs"].shape[0]

    print(f"Processed {n_samples} samples containing {n_atoms} total atoms.")

    assert (
        data_dict["atomic_inputs"].shape[1] == Config.ATOMIC_FEATURE_DIM
    ), f"Atomic feature dim mismatch. Expected {Config.ATOMIC_FEATURE_DIM}, got {data_dict['atomic_inputs'].shape[1]}"
    assert (
        data_dict["global_inputs"].shape[1] == Config.GLOBAL_FEATURE_DIM
    ), f"Global feature dim mismatch. Expected {Config.GLOBAL_FEATURE_DIM}, got {data_dict['global_inputs'].shape[1]}"
    assert data_dict["targets"].shape[0] == n_samples
    assert data_dict["targets"].shape[1] == 2

    # Check scaler generation
    atomic_scaler, global_scaler = get_scalers(data_dict)
    assert atomic_scaler.mean is not None
    assert global_scaler.scale is not None
    print("Scalers fitted successfully.")


def test_data_loading():
    """
    Demonstrates and validates DataLoader creation and batching.
    """
    print("\n--- Testing Data Loading ---")

    train_loader, val_loader = get_train_val_loaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")

    # Validate batch tensors
    atomic_feats = batch["atomic_feats"]
    batch_indices = batch["batch_indices"]
    global_feats = batch["global_feats"]
    targets = batch["targets"]
    ids = batch["ids"]

    print(f"Atomic feats shape: {atomic_feats.shape}")
    print(f"Batch indices shape: {batch_indices.shape}")
    print(f"Global feats shape: {global_feats.shape}")
    print(f"Targets shape: {targets.shape}")

    assert atomic_feats.ndim == 2
    assert atomic_feats.shape[1] == Config.ATOMIC_FEATURE_DIM
    assert global_feats.shape[0] == Config.BATCH_SIZE
    assert global_feats.shape[1] == Config.GLOBAL_FEATURE_DIM
    assert targets.shape == (Config.BATCH_SIZE, 2)
    assert ids.shape == (Config.BATCH_SIZE,)

    # Check batch indices validity
    assert batch_indices.max() < Config.BATCH_SIZE
    assert batch_indices.min() >= 0

    return batch


def test_model_architecture(sample_batch):
    """
    Demonstrates model instantiation and a forward pass.
    """
    print("\n--- Testing Model Architecture ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRRD_DeepSets().to(device)
    print(f"Model instantiated on {device}.")

    # Move batch to device
    atomic_feats = sample_batch["atomic_feats"].to(device)
    batch_indices = sample_batch["batch_indices"].to(device)
    global_feats = sample_batch["global_feats"].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(atomic_feats, batch_indices, global_feats)

    print(f"Model output shape: {output.shape}")

    assert output.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, 2), got {output.shape}"

    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print("Forward pass successful.")


def test_training_loop():
    """
    Demonstrates the training process using the engine.
    """
    print("\n--- Testing Training Loop ---")

    # Run fit function
    # Note: fit() uses Config parameters which we overrode in setup_demo_environment
    model = fit(load_cached_data=True)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print("Training loop completed successfully.")
    return model


def test_inference(model):
    """
    Demonstrates inference on the test set.
    """
    print("\n--- Testing Inference ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader(load_cached_data=False, batch_size=Config.BATCH_SIZE)

    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_feats = batch["atomic_feats"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_feats = batch["global_feats"].to(device)
            batch_ids = batch["ids"]

            # Forward pass (log space)
            outputs = model(atomic_feats, batch_indices, global_feats)

            # Inverse transform (exp space)
            preds_linear = torch.expm1(outputs)

            predictions.append(preds_linear.cpu().numpy())
            ids.append(batch_ids.numpy())

    predictions = np.vstack(predictions)
    ids = np.concatenate(ids)

    print(f"Generated predictions for {len(predictions)} test samples.")

    # Create submission dataframe
    sub_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Save submission
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(sub_df.head())


if __name__ == "__main__":
    # 1. Initialize
    seed_everything(42)
    setup_demo_environment()

    # 2. Test Data Processing
    test_data_processing()

    # 3. Test Data Loading
    sample_batch = test_data_loading()

    # 4. Test Model
    test_model_architecture(sample_batch)

    # 5. Test Training
    trained_model = test_training_loop()

    # 6. Test Inference
    test_inference(trained_model)

    print("\nAll demonstrations completed successfully.")
