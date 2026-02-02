import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import (
    parse_xyz,
    calculate_cell_volume,
    get_chemical_neighbor_distances,
)
from library.data import process_data, CrystalDataset, collate_crystals
from library.model import CRNDSModel
from library.train import train_one_epoch, validate
from library.inference import predict


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting CRN-DS Library Demonstration...")
    set_seed()

    # ==========================================
    # 1. Setup Demo Configuration
    # ==========================================
    # We override the global Config class to use a temporary working directory
    # and smaller hyperparameters for a fast demo run.

    DEMO_DIR = "./working/demo_execution"
    DEMO_CACHE_DIR = "./working/demo_cache"
    DEMO_SUBMISSION_DIR = "./working/demo_submission"

    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_CACHE_DIR, "train_data.npz")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_CACHE_DIR, "val_data.npz")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_CACHE_DIR, "test_data.npz")
    Config.SCALERS_CACHE_PATH = os.path.join(DEMO_DIR, "scalers.npz")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "demo_submission.csv")

    # Override Hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.ATOMIC_HIDDEN_DIM = 32
    Config.ATOMIC_LATENT_DIM = 16
    Config.GLOBAL_HIDDEN_DIM = 16
    Config.GLOBAL_LATENT_DIM = 8
    Config.FUSION_HIDDEN_DIM = 16

    print("Configuration updated for demo execution.")

    # ==========================================
    # 2. Prepare Mini Datasets
    # ==========================================
    # We sample a few rows from the actual metadata to create mini CSVs.
    # This ensures we point to valid geometry files in ./input.

    print("\nPreparing mini datasets...")

    # Load real metadata
    real_train = pd.read_csv("./metadata/train.csv")
    real_val = pd.read_csv("./metadata/val.csv")
    real_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets (e.g., 50 train, 10 val, 10 test)
    mini_train = real_train.head(50).copy()
    mini_val = real_val.head(10).copy()
    mini_test = real_test.head(10).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update Config to point to mini metadata
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path
    Config.TEST_META_PATH = mini_test_path

    print(
        f"Mini datasets created: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
    )

    # ==========================================
    # 3. Test Data Processing & Dataset Class
    # ==========================================
    print("\nTesting Data Processing...")

    # Instantiate CrystalDataset (this triggers process_data internally)
    # We force load_cached_data=False to ensure processing logic runs
    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_META_PATH,
        cache_path=Config.TRAIN_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="train",
        load_cached_data=False,
    )

    # Verify dataset properties
    assert len(train_dataset) == 50, f"Expected 50 samples, got {len(train_dataset)}"
    sample = train_dataset[0]

    print("Sample keys:", sample.keys())
    # Check shapes
    # Atomic features: (N_atoms, 11) -> 4 one-hot + 3 coords + 4 neighbor dists
    assert sample["atomic_features"].shape[1] == Config.ATOMIC_FEATURE_DIM
    # Global features: (12,)
    assert sample["global_features"].shape[0] == Config.GLOBAL_FEATURE_DIM
    # Targets: (2,)
    assert sample["target"].shape[0] == Config.NUM_TARGETS

    print("Dataset verification passed.")

    # ==========================================
    # 4. Test DataLoader & Collation
    # ==========================================
    print("\nTesting DataLoader and Collation...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_crystals,
    )

    batch = next(iter(train_loader))

    # Verify batch structure
    # atomic_features should be stacked: (sum_of_atoms_in_batch, 11)
    # global_features: (batch_size, 12)
    # targets: (batch_size, 2)
    # batch_index: (sum_of_atoms_in_batch,)

    print(f"Batch atomic features shape: {batch['atomic_features'].shape}")
    print(f"Batch global features shape: {batch['global_features'].shape}")
    print(f"Batch index shape: {batch['batch_index'].shape}")

    assert batch["global_features"].shape[0] == Config.BATCH_SIZE
    assert batch["targets"].shape[0] == Config.BATCH_SIZE
    assert batch["atomic_features"].shape[0] == batch["batch_index"].shape[0]

    print("DataLoader verification passed.")

    # ==========================================
    # 5. Test Model Architecture
    # ==========================================
    print("\nTesting Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = CRNDSModel().to(device)

    # Move batch to device
    atomic_feats = batch["atomic_features"].to(device)
    global_feats = batch["global_features"].to(device)
    batch_idx = batch["batch_index"].to(device)

    # Forward pass
    outputs = model(atomic_feats, global_feats, batch_idx)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, Config.NUM_TARGETS)

    print("Model forward pass verified.")

    # ==========================================
    # 6. Test Training Loop
    # ==========================================
    print("\nTesting Training Loop...")

    # Initialize Val Dataset
    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_META_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="val",
        load_cached_data=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_crystals,
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train for a few epochs
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save model for inference test
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
    print("Training loop verified and model saved.")

    # ==========================================
    # 7. Test Inference & Submission
    # ==========================================
    print("\nTesting Inference and Submission Generation...")

    # We will use the library's predict function
    # Note: predict() initializes the test dataset internally using Config.TEST_META_PATH
    # We ensure scalers exist (created during train dataset init)

    # Run prediction
    preds, ids = predict(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, device=device
    )

    print(f"Prediction shape: {preds.shape}")
    print(f"Number of IDs: {len(ids)}")

    assert preds.shape == (len(mini_test), Config.NUM_TARGETS)
    assert len(ids) == len(mini_test)

    # Create submission dataframe manually to verify logic
    # Inverse transform: exp(x) - 1
    original_preds = np.expm1(preds)

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": original_preds[:, 0],
            "bandgap_energy_ev": original_preds[:, 1],
        }
    )
    submission_df.sort_values("id", inplace=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Submission head:")
    print(submission_df.head())

    print("\nInference verified.")
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()
