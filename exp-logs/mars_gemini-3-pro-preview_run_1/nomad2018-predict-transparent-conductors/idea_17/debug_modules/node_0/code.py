import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths for the demo
DEMO_DIR = "./working/demo_execution"
os.makedirs(DEMO_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Create Mini Datasets for Speed
# ------------------------------------------------------------------------------
print("[Demo] Creating mini datasets...")

# Load original metadata
train_full = pd.read_csv("./metadata/train.csv")
val_full = pd.read_csv("./metadata/val.csv")
test_full = pd.read_csv("./metadata/test.csv")

# Sample a small subset (e.g., 50 train, 10 val, 10 test)
mini_train = train_full.head(50).copy()
mini_val = val_full.head(10).copy()
mini_test = test_full.head(10).copy()

# Save mini metadata to working directory
mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

mini_train.to_csv(mini_train_path, index=False)
mini_val.to_csv(mini_val_path, index=False)
mini_test.to_csv(mini_test_path, index=False)

print(
    f"Created mini datasets: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
)

# ------------------------------------------------------------------------------
# 2. Patch Library Configuration to Use Mini Datasets
# ------------------------------------------------------------------------------
# We need to modify the paths in the imported modules so they use our mini files
# and output to our demo directory.

import library.config
import library.features
import library.dataset
import library.engine
import library.data_utils

# Patch paths in library.config (referenced by other modules)
library.config.WORKING_DIR = DEMO_DIR
library.config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_data.npz")
library.config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_data.npz")
library.config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_data.npz")
library.config.SCALERS_CACHE_PATH = os.path.join(DEMO_DIR, "scalers.npz")
library.config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pt")

# Patch paths in library.features (since they are imported directly)
library.features.TRAIN_METADATA_PATH = mini_train_path
library.features.VAL_METADATA_PATH = mini_val_path
library.features.TEST_METADATA_PATH = mini_test_path
library.features.TRAIN_CACHE_PATH = library.config.TRAIN_CACHE_PATH
library.features.VAL_CACHE_PATH = library.config.VAL_CACHE_PATH
library.features.TEST_CACHE_PATH = library.config.TEST_CACHE_PATH
library.features.SCALERS_CACHE_PATH = library.config.SCALERS_CACHE_PATH
library.features.WORKING_DIR = DEMO_DIR

# Patch training parameters in library.engine
library.engine.NUM_EPOCHS = 1  # Run only 1 epoch for demo
library.engine.WORKING_DIR = DEMO_DIR
library.engine.MODEL_SAVE_PATH = library.config.MODEL_SAVE_PATH

# Create a demo submission dir
DEMO_SUBMISSION_DIR = "./working/demo_submission"
os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)
library.engine.SUBMISSION_PATH = os.path.join(
    DEMO_SUBMISSION_DIR, "demo_submission.csv"
)

print("[Demo] Configuration patched for speed and isolation.")

# ------------------------------------------------------------------------------
# 3. Verify Data Utilities
# ------------------------------------------------------------------------------
print("\n[Demo] Verifying Data Utilities...")

# Pick a sample file from the mini train set
sample_rel_path = mini_train.iloc[0]["file_path"]
sample_full_path = os.path.join(library.config.INPUT_DIR, sample_rel_path)

print(f"Testing parsing on: {sample_full_path}")
lattice, species, coords = library.data_utils.parse_xyz(sample_full_path)

assert lattice.shape == (3, 3), "Lattice vectors shape mismatch"
assert len(species) == len(
    coords
), "Mismatch between species count and coordinate count"
assert coords.shape[1] == 3, "Coordinates should be 3D"

print("  -> parse_xyz passed.")

# Test atomic features extraction
atomic_feats = library.data_utils.get_atomic_features(sample_full_path)
# Expected dim: 4 (one-hot) + 3 (coords) + 1 (nn) + 1 (potential) = 9
assert (
    atomic_feats.shape[1] == 9
), f"Atomic features dim mismatch. Expected 9, got {atomic_feats.shape[1]}"
assert atomic_feats.shape[0] == len(species), "Atomic features row count mismatch"

print("  -> get_atomic_features passed.")

# Test global features extraction
# Create a dummy row series from the dataframe
sample_row = mini_train.iloc[0]
global_feats = library.data_utils.get_global_features(sample_row, lattice, len(species))
# Expected dim: 12
assert global_feats.shape == (
    12,
), f"Global features shape mismatch. Expected (12,), got {global_feats.shape}"

print("  -> get_global_features passed.")

# ------------------------------------------------------------------------------
# 4. Verify Dataset Preparation
# ------------------------------------------------------------------------------
print("\n[Demo] Verifying Dataset Preparation (this runs the feature pipeline)...")

# Force reload to ensure we process our mini datasets
train_data, val_data, test_data, scalers = library.features.prepare_datasets(
    load_cached_data=False
)

assert os.path.exists(library.config.TRAIN_CACHE_PATH), "Train cache not created"
assert (
    len(train_data["targets"]) == 50
), f"Expected 50 train samples, got {len(train_data['targets'])}"
assert (
    len(val_data["targets"]) == 10
), f"Expected 10 val samples, got {len(val_data['targets'])}"

print("  -> prepare_datasets passed.")

# ------------------------------------------------------------------------------
# 5. Verify Model and DataLoaders
# ------------------------------------------------------------------------------
print("\n[Demo] Verifying Model and DataLoaders...")

# Get dataloaders (will load the cached mini data we just created)
train_loader, val_loader, test_loader = library.dataset.get_dataloaders(
    batch_size=4, num_workers=0
)

# Fetch one batch
atom_x, batch_indices, global_x, targets, ids = next(iter(train_loader))

print(
    f"  Batch shapes: AtomX={atom_x.shape}, BatchIdx={batch_indices.shape}, GlobalX={global_x.shape}, Targets={targets.shape}"
)

# Instantiate model
model = library.model.PCWDSModel()
# Move to CPU for this quick test if CUDA is not mandatory, but config uses auto-detect
device = library.config.DEVICE
model.to(device)

# Forward pass
atom_x = atom_x.to(device)
batch_indices = batch_indices.to(device)
global_x = global_x.to(device)

outputs = model(atom_x, batch_indices, global_x)

assert outputs.shape == (
    4,
    2,
), f"Output shape mismatch. Expected (4, 2), got {outputs.shape}"
print("  -> Model forward pass passed.")

# ------------------------------------------------------------------------------
# 6. Verify Training Loop
# ------------------------------------------------------------------------------
print("\n[Demo] Verifying Training Loop...")

# We use the engine's train_one_epoch function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Run one epoch
loss = library.engine.train_one_epoch(model, train_loader, optimizer, criterion, device)
print(f"  -> Train One Epoch Loss: {loss:.4f}")

# Run validation
val_loss, val_rmsle = library.engine.validate(model, val_loader, criterion, device)
print(f"  -> Validation Loss: {val_loss:.4f}, RMSLE: {val_rmsle:.4f}")

# Save this model as "best_model.pt" so generate_submission can find it
torch.save(model.state_dict(), library.config.MODEL_SAVE_PATH)
print("  -> Dummy model saved for submission generation.")

# ------------------------------------------------------------------------------
# 7. Verify Submission Generation
# ------------------------------------------------------------------------------
print("\n[Demo] Verifying Submission Generation...")

# Run the generation function
library.engine.generate_submission(load_cached_data=True)

assert os.path.exists(library.engine.SUBMISSION_PATH), "Submission file not created"
df_sub = pd.read_csv(library.engine.SUBMISSION_PATH)
assert len(df_sub) == 10, f"Expected 10 predictions in mini-test, got {len(df_sub)}"
assert "formation_energy_ev_natom" in df_sub.columns
assert "bandgap_energy_ev" in df_sub.columns

print("  -> Submission generation passed.")

print("\n[Demo] All verifications passed successfully!")
