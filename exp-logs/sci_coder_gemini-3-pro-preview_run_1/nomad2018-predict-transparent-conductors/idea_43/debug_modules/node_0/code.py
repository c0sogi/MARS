import os
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library
from library.config import Config
from library.data import get_data_loaders
from library.model import AMSP_DS_Net
from library.train_eval import Trainer


def run_demo():
    print("Initializing AMSP-DS Demo...")

    # 1. Setup Directories for Demo
    # We use the working directory to avoid messing with original metadata/cache if possible,
    # though the prompt says metadata is read-only essentially (pre-generated).
    # We will create a separate config setup for the demo.

    demo_base = "./working"
    demo_meta_dir = os.path.join(demo_base, "demo_metadata")
    demo_exec_dir = os.path.join(demo_base, "demo_execution")
    demo_sub_dir = os.path.join(demo_base, "demo_submission")

    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_exec_dir, exist_ok=True)
    os.makedirs(demo_sub_dir, exist_ok=True)

    # 2. Create Subset Metadata
    # We take a small slice of the provided metadata to ensure the demo runs quickly.
    print("Creating subset metadata for demonstration...")

    # Read original metadata
    orig_train = pd.read_csv(Config.TRAIN_METADATA)
    orig_val = pd.read_csv(Config.VAL_METADATA)
    orig_test = pd.read_csv(Config.TEST_METADATA)

    # Subset (e.g., 50 samples each)
    subset_size = 50
    demo_train = orig_train.head(subset_size)
    demo_val = orig_val.head(subset_size)
    demo_test = orig_test.head(subset_size)

    # Save to demo directory
    demo_train_path = os.path.join(demo_meta_dir, "train.csv")
    demo_val_path = os.path.join(demo_meta_dir, "val.csv")
    demo_test_path = os.path.join(demo_meta_dir, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(f"Created subset metadata with {subset_size} samples each.")

    # 3. Modify Config for Demo
    # We modify the Config class attributes directly to point to our demo files
    print("Configuring parameters for speed...")
    Config.TRAIN_METADATA = demo_train_path
    Config.VAL_METADATA = demo_val_path
    Config.TEST_METADATA = demo_test_path

    Config.WORKING_DIR = demo_exec_dir
    Config.TRAIN_DATA_CACHE = os.path.join(demo_exec_dir, "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(demo_exec_dir, "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(demo_exec_dir, "test_data.npz")
    Config.SCALERS_CACHE = os.path.join(demo_exec_dir, "scalers.npz")
    Config.MODEL_CHECKPOINT = os.path.join(demo_exec_dir, "best_model.pt")

    Config.SUBMISSION_PATH = os.path.join(demo_sub_dir, "demo_submission.csv")

    # Reduce training parameters
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 16

    # 4. Data Loading and Processing
    print("\n[Step 1] Loading and Processing Data...")
    # We set load_cached=False to demonstrate the processing pipeline logic
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        load_cached=False,
        num_workers=0,  # Use 0 workers for simple script execution to avoid multiprocessing overhead in demo
    )

    # Verify DataLoaders
    print("Verifying DataLoader batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = {
        "atomic_features",
        "batch_indices",
        "global_features",
        "targets",
        "ids",
    }
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Check dimensions
    # atomic_features: (N_atoms, atom_dim)
    # global_features: (Batch, global_dim)
    # targets: (Batch, 2)

    atom_dim = batch["atomic_features"].shape[1]
    global_dim = batch["global_features"].shape[1]
    batch_size = batch["targets"].shape[0]

    print(f"  Batch Size: {batch_size}")
    print(f"  Atomic Feature Dim: {atom_dim}")
    print(f"  Global Feature Dim: {global_dim}")

    assert (
        batch_size == Config.BATCH_SIZE or batch_size == subset_size % Config.BATCH_SIZE
    ), "Unexpected batch size"
    assert batch["targets"].shape[1] == 2, "Targets should have 2 columns"

    # 5. Model Instantiation
    print("\n[Step 2] Instantiating Model...")
    model = AMSP_DS_Net(
        atom_input_dim=atom_dim,
        global_input_dim=global_dim,
        atomic_hidden_dim=64,  # Reduced for demo speed
        atomic_layers=2,
        global_hidden_dim=32,
        global_layers=2,
        fusion_hidden_dim=32,
        dropout_rate=0.1,
    )

    # Verify Forward Pass
    print("Verifying Forward Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    atomic_feats = batch["atomic_features"].to(device)
    batch_idx = batch["batch_indices"].to(device)
    global_feats = batch["global_features"].to(device)

    with torch.no_grad():
        output = model(atomic_feats, batch_idx, global_feats)

    assert output.shape == (
        batch_size,
        2,
    ), f"Output shape mismatch. Expected ({batch_size}, 2), got {output.shape}"
    print("  Forward pass successful.")

    # 6. Training
    print("\n[Step 3] Running Training Loop...")
    trainer = Trainer(model, device=device)
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Verify checkpoint creation
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created."
    print("  Training loop completed and model saved.")

    # 7. Inference
    print("\n[Step 4] Running Inference...")
    ids, preds = trainer.predict(test_loader)

    assert (
        len(ids) == subset_size
    ), f"Expected {subset_size} predictions, got {len(ids)}"
    assert preds.shape == (
        subset_size,
        2,
    ), f"Prediction shape mismatch. Got {preds.shape}"

    # 8. Saving Submission
    print("\n[Step 5] Generating Submission...")
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )
    submission_df.sort_values("id", inplace=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    print(f"  Submission saved to {Config.SUBMISSION_PATH}")
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
