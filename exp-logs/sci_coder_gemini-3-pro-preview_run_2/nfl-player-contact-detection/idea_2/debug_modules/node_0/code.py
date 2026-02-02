import os
import pandas as pd
import numpy as np
import torch
import shutil
import sys

# =============================================================================
# 1. DATA SUBSET CREATION
# =============================================================================
# We create a small subset of the data to ensure the demonstration runs quickly.
# This involves reading the first few plays from metadata and filtering the
# tracking data accordingly.

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
DEMO_DIR = os.path.join(WORKING_DIR, "demo_data")
os.makedirs(DEMO_DIR, exist_ok=True)

print("--- Preparing Subset Data for Demonstration ---")

# Load original metadata
train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
val_meta = pd.read_csv(os.path.join(METADATA_DIR, "validation.csv"))

# Select a small number of game_plays (2 for train, 1 for val)
train_plays = train_meta["game_play"].unique()[:2]
val_plays = val_meta["game_play"].unique()[:1]

subset_train_meta = train_meta[train_meta["game_play"].isin(train_plays)].copy()
subset_val_meta = val_meta[val_meta["game_play"].isin(val_plays)].copy()

# Save subset metadata
demo_train_meta_path = os.path.join(DEMO_DIR, "train_meta.csv")
demo_val_meta_path = os.path.join(DEMO_DIR, "val_meta.csv")
subset_train_meta.to_csv(demo_train_meta_path, index=False)
subset_val_meta.to_csv(demo_val_meta_path, index=False)

print(
    f"Subset Metadata Created: {len(subset_train_meta)} train rows, {len(subset_val_meta)} val rows."
)

# Load and filter tracking data
# We read the full file but filter immediately to keep the demo file small.
print("Filtering tracking data (this may take a moment)...")
train_tracking = pd.read_csv(os.path.join(INPUT_DIR, "train_player_tracking.csv"))
all_demo_plays = np.concatenate([train_plays, val_plays])
subset_tracking = train_tracking[
    train_tracking["game_play"].isin(all_demo_plays)
].copy()

demo_tracking_path = os.path.join(DEMO_DIR, "train_tracking.csv")
subset_tracking.to_csv(demo_tracking_path, index=False)
print("Subset Tracking Data Created.")

# =============================================================================
# 2. CONFIGURATION PATCHING
# =============================================================================
# We must import library.config and patch the paths BEFORE importing other
# library modules. This ensures they load the modified configuration.

import library.config as config

print("\n--- Patching Library Configuration ---")
config.TRAIN_META_PATH = demo_train_meta_path
config.VAL_META_PATH = demo_val_meta_path
# Use the same tracking file for both; the FeatureEngineer filters by game_play anyway
config.TRAIN_TRACKING_PATH = demo_tracking_path
config.WORKING_DIR = os.path.join(WORKING_DIR, "demo_working")

# Update derived paths in config to ensure consistency
os.makedirs(config.WORKING_DIR, exist_ok=True)
config.CACHE_TRAIN_FEATURES = os.path.join(
    config.WORKING_DIR, "train_features_seq.parquet"
)
config.CACHE_VAL_FEATURES = os.path.join(config.WORKING_DIR, "val_features_seq.parquet")
config.CACHE_TEST_FEATURES = os.path.join(
    config.WORKING_DIR, "test_features_seq.parquet"
)
config.CACHE_SCALER = os.path.join(config.WORKING_DIR, "scaler.joblib")
config.MODEL_CHECKPOINT_PATH = os.path.join(config.WORKING_DIR, "dstcn_model.pth")

# Optimize hyperparameters for speed
config.EPOCHS = 2
config.BATCH_SIZE = 32
config.PATIENCE = 1

# =============================================================================
# 3. IMPORT LIBRARY MODULES
# =============================================================================
from library.feature_engineering import FeatureEngineer
from library.dataset import NFLSequenceDataset
from library.trainer import Trainer
from library.utils import seed_everything
from library.model import DualStreamTCN

# Set random seed for reproducibility
seed_everything(42)

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n--- Running Feature Engineering ---")
fe = FeatureEngineer()

# Process Training Data
# load_cached_data=False forces regeneration using our new subset files
print("Generating Training Features...")
X_train, y_train, ids_train = fe.process_train(load_cached_data=False)

# Validation
print(f"X_train shape: {X_train.shape}")
assert len(X_train) == len(subset_train_meta), "X_train size does not match metadata"
assert (
    X_train.shape[1] == config.WINDOW_SIZE
), f"Expected window size {config.WINDOW_SIZE}"
assert not np.isnan(X_train).any(), "X_train contains NaNs"

# Process Validation Data
print("Generating Validation Features...")
X_val, y_val, ids_val = fe.process_val(load_cached_data=False)
print(f"X_val shape: {X_val.shape}")

# =============================================================================
# 5. DATASET & MODEL VERIFICATION
# =============================================================================
print("\n--- Verifying Dataset and Model ---")

# Test Dataset Class
ds_train = NFLSequenceDataset(X_train, y_train)
sample = ds_train[0]

print("Dataset Sample Keys:", sample.keys())
assert "features" in sample
assert "is_ground" in sample
assert "target" in sample
assert sample["features"].shape == (config.WINDOW_SIZE, len(config.FEATURE_COLS))
# is_ground should be a scalar (0-d tensor) or 1-d tensor of size 1
assert sample["is_ground"].numel() == 1

# Test Model Architecture
model = DualStreamTCN()
# Create a small batch
batch_features = torch.stack([ds_train[i]["features"] for i in range(4)])
batch_is_ground = torch.stack([ds_train[i]["is_ground"] for i in range(4)])

print(f"Model Input Shape: {batch_features.shape}")
with torch.no_grad():
    logits = model(batch_features, batch_is_ground)

print(f"Model Output Shape: {logits.shape}")
assert logits.shape == (4, 1), "Model output shape mismatch"

# =============================================================================
# 6. TRAINING
# =============================================================================
print("\n--- Starting Training Loop ---")
trainer = Trainer()

# Fit the model
trainer.fit(X_train, y_train, X_val, y_val)

# Check if checkpoint was created
if os.path.exists(config.MODEL_CHECKPOINT_PATH):
    print(f"Model checkpoint successfully saved at {config.MODEL_CHECKPOINT_PATH}")
else:
    print("Warning: No checkpoint found (possibly no improvement in validation).")

# =============================================================================
# 7. INFERENCE
# =============================================================================
print("\n--- Running Inference ---")
# Predict on validation set
preds = trainer.predict(X_val)

print(f"Predictions Shape: {preds.shape}")
print(f"Sample Predictions: {preds[:5]}")

# Validation
assert len(preds) == len(y_val), "Prediction length mismatch"
assert np.all((preds >= 0) & (preds <= 1)), "Predictions outside [0,1] range"

print("\n=== Demonstration Completed Successfully ===")
