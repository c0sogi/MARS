import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import get_datasets, collate_fn
from library.model import APDeepSets
from library.train import train_model
from library.predict import generate_submission


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting Library Demo...")
    set_seed()

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    # We override Config paths to use a separate demo directory within ./working
    # to avoid interfering with any production runs.
    DEMO_DIR = "./working/demo_execution"
    DEMO_SUBMISSION_DIR = "./working/demo_submission"

    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    if os.path.exists(DEMO_SUBMISSION_DIR):
        shutil.rmtree(DEMO_SUBMISSION_DIR)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print(f"Setting up demo working directory: {DEMO_DIR}")

    # Monkey-patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_data.npz")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "demo_model.pt")
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "demo_submission.csv")

    # Reduce hyperparameters for speed
    Config.HIDDEN_DIM = 64
    Config.LATENT_DIM = 64
    Config.BATCH_SIZE = 4

    # -------------------------------------------------------------------------
    # 2. Data Loading and Preprocessing Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loading ---")
    # We use debug=True and max_samples=20 to keep it very fast
    train_ds, val_ds, test_ds = get_datasets(debug=True, max_samples=20)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    assert len(train_ds) <= 20, "Train dataset should be limited by max_samples"
    assert len(val_ds) <= 20, "Val dataset should be limited by max_samples"

    # Check a single sample
    sample = train_ds[0]
    print("Sample keys:", sample.keys())
    assert "id" in sample
    assert "global_features" in sample
    assert "atomic_features" in sample
    assert "targets" in sample

    # Check feature dimensions
    # Global: 12 features
    assert (
        sample["global_features"].shape[0] == 12
    ), f"Expected 12 global features, got {sample['global_features'].shape[0]}"
    # Atomic: N_atoms x 8 features
    assert sample["atomic_features"].ndim == 2
    assert (
        sample["atomic_features"].shape[1] == 8
    ), f"Expected 8 atomic features, got {sample['atomic_features'].shape[1]}"

    print("Data loading verified.")

    # -------------------------------------------------------------------------
    # 3. Dataloader and Collate Function Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Collate Function ---")
    loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn, shuffle=False)
    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    # Check batch structure
    # Global features should be stacked: (B, 12)
    assert batch["global_features"].shape == (
        4,
        12,
    ), "Global features batch shape mismatch"
    # Atomic features should be concatenated: (Sum_Atoms, 8)
    total_atoms = batch["atomic_features"].shape[0]
    assert (
        batch["atomic_features"].shape[1] == 8
    ), "Atomic features feature dim mismatch"
    # Batch indices should match total atoms
    assert (
        batch["batch_indices"].shape[0] == total_atoms
    ), "Batch indices shape mismatch"
    # Targets should be stacked: (B, 2)
    assert batch["targets"].shape == (4, 2), "Targets batch shape mismatch"

    print("Collate function verified.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Testing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APDeepSets()
    model.to(device)

    # Move batch to device
    model_input = {
        "global_features": batch["global_features"].to(device),
        "atomic_features": batch["atomic_features"].to(device),
        "batch_indices": batch["batch_indices"].to(device),
    }

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(model_input)

    print("Model output shape:", output.shape)
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Training Loop ---")
    # Train for 2 epochs on the small subset
    try:
        train_model(debug=True, max_samples=20, num_epochs=2)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify model checkpoint was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint successfully saved at {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # -------------------------------------------------------------------------
    # 6. Prediction Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Prediction ---")
    try:
        generate_submission(debug=True, max_samples=10)
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        raise e

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file successfully created at {Config.SUBMISSION_PATH}")
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df.shape}")
        assert (
            df.shape[1] == 3
        ), "Submission should have 3 columns (id, formation_energy, bandgap)"
        assert len(df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
