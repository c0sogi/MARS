import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import CRR_DS_Model
from library.engine import Trainer


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing demonstration...")
    set_seed(Config.SEED)

    # 1. Override Config for Speed
    # We want the demo to run quickly, so we reduce epochs and batch size
    print("Overriding configuration for speed...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    # Ensure working directories exist (Config.setup_directories called on import, but good to be safe)
    Config.setup_directories()

    # 2. Test Data Loading Pipeline
    print("\nTesting Data Loading Pipeline...")
    # We force reprocessing to demonstrate the feature extraction logic.
    # The dataset size is small enough (~2000 samples) to process quickly.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify a batch
    sample_batch = next(iter(train_loader))
    # print("Sample batch keys:", sample_batch.keys())

    atomic_feats = sample_batch["atomic_features"]
    global_feats = sample_batch["global_features"]
    batch_indices = sample_batch["batch_indices"]
    targets = sample_batch["targets"]

    print(f"Atomic features shape: {atomic_feats.shape}")
    print(f"Global features shape: {global_feats.shape}")
    print(f"Batch indices shape: {batch_indices.shape}")
    print(f"Targets shape: {targets.shape}")

    # Assertions for data shapes
    # Atomic features dim is 12 (4 one-hot + 3 coords + 4 recip + 1 density)
    assert (
        atomic_feats.shape[1] == Config.ATOMIC_INPUT_DIM
    ), f"Expected {Config.ATOMIC_INPUT_DIM} atomic features, got {atomic_feats.shape[1]}"
    # Global features dim is 12 (3 lat len + 3 lat ang + 1 vol + 1 dens + 3 stoich + 1 total)
    assert (
        global_feats.shape[1] == Config.GLOBAL_INPUT_DIM
    ), f"Expected {Config.GLOBAL_INPUT_DIM} global features, got {global_feats.shape[1]}"
    # Targets dim is 2
    assert targets.shape[1] == 2, "Expected 2 targets"

    # 3. Test Model Architecture
    print("\nTesting Model Architecture...")
    model = CRR_DS_Model().to(Config.DEVICE)

    # Move batch to device
    atomic_feats = atomic_feats.to(Config.DEVICE)
    global_feats = global_feats.to(Config.DEVICE)
    batch_indices = batch_indices.to(Config.DEVICE)

    # Forward pass
    outputs = model(atomic_feats, global_feats, batch_indices)
    print(f"Model output shape: {outputs.shape}")

    assert outputs.shape == (global_feats.shape[0], 2), "Output shape mismatch"

    # 4. Test Training Loop
    print("\nTesting Training Loop...")
    trainer = Trainer(device=torch.device(Config.DEVICE))

    # Run fit (with reduced epochs from override)
    trainer.fit(epochs=Config.EPOCHS, patience=1)

    # Check if model checkpoint exists
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Model checkpoint not created after training.")
    print("Training complete and model saved.")

    # 5. Test Inference and Submission Generation
    print("\nTesting Inference and Submission...")
    trainer.generate_submission()

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file not created.")

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    # print("Submission columns:", sub_df.columns.tolist())

    assert sub_df.shape[1] == 3, "Submission should have 3 columns"
    assert "id" in sub_df.columns
    assert "formation_energy_ev_natom" in sub_df.columns
    assert "bandgap_energy_ev" in sub_df.columns
    assert len(sub_df) > 0, "Submission is empty"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
