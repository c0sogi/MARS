import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.features import parse_xyz, get_local_features, get_global_features
from library.data import CrystalDataset, collate_sparse_batch
from library.model import GBAMSDSModel
from library.train import train_epoch, validate


def main():
    print("Starting GBA-MS-DS Library Demonstration...")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override for Demo
    # ---------------------------------------------------------
    print("\n[1] Setting up configuration for fast demonstration...")

    # Define demo-specific paths to avoid conflicts with main runs
    DEMO_DIR = "./working/demo_execution"
    DEMO_META_DIR = "./working/demo_metadata"
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE_FILE = os.path.join(DEMO_DIR, "train_data.npz")
    Config.VAL_CACHE_FILE = os.path.join(DEMO_DIR, "val_data.npz")
    Config.TEST_CACHE_FILE = os.path.join(DEMO_DIR, "test_data.npz")
    Config.SCALER_CACHE_FILE = os.path.join(DEMO_DIR, "scalers.npz")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pt")

    # Reduce model size and training duration for the demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.ATOMIC_HIDDEN_DIM = 32
    Config.GLOBAL_HIDDEN_DIM = 16
    Config.FUSION_HIDDEN_DIM = 16

    # Create subset metadata files to process only a few samples
    # We read the original metadata and take the top N rows
    full_train_df = pd.read_csv("./metadata/train.csv")
    full_val_df = pd.read_csv("./metadata/val.csv")
    full_test_df = pd.read_csv("./metadata/test.csv")

    # Subset size: 20 train, 10 val, 10 test
    demo_train_df = full_train_df.head(20).copy()
    demo_val_df = full_val_df.head(10).copy()
    demo_test_df = full_test_df.head(10).copy()

    demo_train_path = os.path.join(DEMO_META_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_META_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_META_DIR, "test.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    # Point Config to these new metadata files
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    print("   Configuration updated and subset metadata created.")

    # ---------------------------------------------------------
    # 2. Feature Extraction Demonstration
    # ---------------------------------------------------------
    print("\n[2] Demonstrating Feature Extraction...")

    # Pick one file to demonstrate low-level functions
    sample_row = demo_train_df.iloc[0]
    sample_file_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    # Test parse_xyz
    atom_types, coords, lattice_vectors = parse_xyz(sample_file_path)
    print(f"   Parsed {len(atom_types)} atoms from {sample_row['file_path']}")

    # Verify parsing
    assert len(atom_types) > 0, "No atoms found in file"
    assert coords.shape == (len(atom_types), 3), "Incorrect coordinates shape"
    assert lattice_vectors.shape == (3, 3), "Incorrect lattice vectors shape"

    # Test get_local_features
    local_feats = get_local_features(atom_types, coords, lattice_vectors)
    print(f"   Local features shape: {local_feats.shape}")
    # Expected: (N_atoms, ATOMIC_FEATURE_DIM)
    assert (
        local_feats.shape[1] == Config.ATOMIC_FEATURE_DIM
    ), f"Expected atomic feature dim {Config.ATOMIC_FEATURE_DIM}, got {local_feats.shape[1]}"

    # Test get_global_features
    global_feats = get_global_features(atom_types, coords, lattice_vectors)
    print(f"   Global features shape: {global_feats.shape}")
    # Expected: (GLOBAL_FEATURE_DIM,)
    assert (
        global_feats.shape[0] == Config.GLOBAL_FEATURE_DIM
    ), f"Expected global feature dim {Config.GLOBAL_FEATURE_DIM}, got {global_feats.shape[0]}"

    # ---------------------------------------------------------
    # 3. Data Processing & Dataset Demonstration
    # ---------------------------------------------------------
    print("\n[3] Demonstrating Dataset Processing...")

    # Ensure we process from scratch for the demo
    if os.path.exists(Config.TRAIN_CACHE_FILE):
        os.remove(Config.TRAIN_CACHE_FILE)

    # Initialize Training Dataset
    # This triggers process_data -> get_scalers -> scaling -> target transform
    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_file=Config.TRAIN_CACHE_FILE,
        fit_scalers=True,
        transform_targets=True,
        load_cached_data=False,
    )

    print(f"   Train dataset size: {len(train_dataset)}")
    assert len(train_dataset) == 20, "Dataset size mismatch"

    # Check __getitem__ output
    atoms_t, glob_t, target_t, id_t = train_dataset[0]
    print(
        f"   Sample 0 shapes: Atoms {atoms_t.shape}, Global {glob_t.shape}, Target {target_t.shape}"
    )

    assert atoms_t.ndim == 2 and atoms_t.shape[1] == Config.ATOMIC_FEATURE_DIM
    assert glob_t.shape[0] == Config.GLOBAL_FEATURE_DIM
    assert target_t.shape[0] == 2

    # Initialize Validation Dataset (using scalers fitted on train)
    if os.path.exists(Config.VAL_CACHE_FILE):
        os.remove(Config.VAL_CACHE_FILE)

    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_file=Config.VAL_CACHE_FILE,
        scalers=(train_dataset.a_scaler, train_dataset.g_scaler),
        fit_scalers=False,
        transform_targets=True,
        load_cached_data=False,
    )
    print(f"   Val dataset size: {len(val_dataset)}")

    # ---------------------------------------------------------
    # 4. DataLoader & Batching Demonstration
    # ---------------------------------------------------------
    print("\n[4] Demonstrating DataLoader & Sparse Batching...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sparse_batch,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print("   Batch keys:", list(batch.keys()))
    print(f"   Batch atomic features shape: {batch['atomic_features'].shape}")
    print(f"   Batch global features shape: {batch['global_features'].shape}")
    print(f"   Batch indices shape: {batch['batch_indices'].shape}")

    # Verify batch structure
    # Global features should have batch_size rows
    assert batch["global_features"].shape[0] == Config.BATCH_SIZE
    # Batch indices should map every atom to a graph index
    assert batch["atomic_features"].shape[0] == batch["batch_indices"].shape[0]
    # Max batch index should be less than batch size
    assert batch["batch_indices"].max() < Config.BATCH_SIZE

    # ---------------------------------------------------------
    # 5. Model Demonstration
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Model Forward Pass...")

    device = torch.device("cpu")  # Use CPU for demo to ensure compatibility
    model = GBAMSDSModel().to(device)

    # Move batch to device
    inputs = {
        "atomic_features": batch["atomic_features"].to(device),
        "global_features": batch["global_features"].to(device),
        "batch_indices": batch["batch_indices"].to(device),
    }

    # Run forward pass
    outputs = model(inputs)
    print(f"   Model output shape: {outputs.shape}")

    # Verify output shape (Batch_Size, 2)
    assert outputs.shape == (Config.BATCH_SIZE, 2)

    # ---------------------------------------------------------
    # 6. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Training Loop...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Run for configured small epochs
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # Validate (using train_loader here just to show functionality on small data)
        val_loss, val_rmsle_f, val_rmsle_b = validate(
            model, train_loader, criterion, device
        )
        print(
            f"   Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"   Model saved to {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 7. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[7] Demonstrating Inference...")

    if os.path.exists(Config.TEST_CACHE_FILE):
        os.remove(Config.TEST_CACHE_FILE)

    # Load Test Data using scalers from training
    test_dataset = CrystalDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_file=Config.TEST_CACHE_FILE,
        scalers=(train_dataset.a_scaler, train_dataset.g_scaler),
        fit_scalers=False,
        transform_targets=False,  # No targets in test
        load_cached_data=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse_batch,
    )

    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "atomic_features": batch["atomic_features"].to(device),
                "global_features": batch["global_features"].to(device),
                "batch_indices": batch["batch_indices"].to(device),
            }
            # Forward pass (model outputs are in log space because we trained on log targets)
            out_log = model(inputs)

            # Inverse transform: exp(x) - 1
            out_real = torch.expm1(out_log)

            all_ids.append(batch["ids"])
            all_preds.append(out_real)

    all_ids = torch.cat(all_ids).numpy()
    all_preds = torch.cat(all_preds).numpy()

    print(f"   Inference complete. Predictions shape: {all_preds.shape}")
    assert len(all_ids) == 10  # We used head(10) for test

    # Create submission dataframe
    sub_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    print("   Sample predictions:")
    print(sub_df.head(3))

    # Cleanup demo directory
    # shutil.rmtree(DEMO_DIR)
    # shutil.rmtree(DEMO_META_DIR)

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
