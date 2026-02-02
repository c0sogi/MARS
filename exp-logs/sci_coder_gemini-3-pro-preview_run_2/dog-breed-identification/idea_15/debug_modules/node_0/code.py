import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

# Import provided library modules
import library.config
import library.dataset
import library.model_factory
import library.feature_engine
import library.training
import library.ensemble

# =============================================================================
# 1. Configuration and Setup
# =============================================================================
DEMO_WORKING_DIR = "./working/demo_run"
DEMO_METADATA_DIR = "./working/demo_metadata"
DEMO_SUBMISSION_DIR = "./working/demo_submission"
DEMO_SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

print("=== Setting up Demo Environment ===")

# =============================================================================
# 2. Create Minimal Dataset (Subset)
# =============================================================================
# We need at least one sample per class for the training set to ensure the
# classifier learns 120 classes and outputs the correct shape.
print("Creating dataset subsets...")

# Load original metadata
orig_train_df = pd.read_csv(library.config.TRAIN_METADATA_PATH)
orig_val_df = pd.read_csv(library.config.VAL_METADATA_PATH)
orig_test_df = pd.read_csv(library.config.TEST_METADATA_PATH)

# Create stratified train subset (1 per breed) to minimize size but keep all classes
train_subset = orig_train_df.groupby("breed").head(1).reset_index(drop=True)
# Create small val and test subsets
val_subset = orig_val_df.sample(n=20, random_state=42).reset_index(drop=True)
test_subset = orig_test_df.sample(n=20, random_state=42).reset_index(drop=True)

# Save to demo location
demo_train_path = os.path.join(DEMO_METADATA_DIR, "train.csv")
demo_val_path = os.path.join(DEMO_METADATA_DIR, "val.csv")
demo_test_path = os.path.join(DEMO_METADATA_DIR, "test.csv")

train_subset.to_csv(demo_train_path, index=False)
val_subset.to_csv(demo_val_path, index=False)
test_subset.to_csv(demo_test_path, index=False)

print(f"Train subset: {len(train_subset)} samples (1 per class)")
print(f"Val subset:   {len(val_subset)} samples")
print(f"Test subset:  {len(test_subset)} samples")

# =============================================================================
# 3. Patch Library Configurations
# =============================================================================
print("Patching library configurations...")

# Patch Config Paths
library.config.TRAIN_METADATA_PATH = demo_train_path
library.config.VAL_METADATA_PATH = demo_val_path
library.config.TEST_METADATA_PATH = demo_test_path
library.config.WORKING_DIR = DEMO_WORKING_DIR
library.config.SUBMISSION_PATH = DEMO_SUBMISSION_PATH

# Patch Dataset Module Paths (Since they were imported as constants)
library.dataset.TRAIN_METADATA_PATH = demo_train_path
library.dataset.VAL_METADATA_PATH = demo_val_path
library.dataset.TEST_METADATA_PATH = demo_test_path

# Patch Feature Engine and Training Modules Working Directory
library.feature_engine.WORKING_DIR = DEMO_WORKING_DIR
library.training.WORKING_DIR = DEMO_WORKING_DIR
library.training.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
library.ensemble.SUBMISSION_PATH = DEMO_SUBMISSION_PATH

# =============================================================================
# 4. Mock LogisticRegressionCV
# =============================================================================
# Standard LogisticRegressionCV fails with cv=5 on a dataset with 1 sample per class.
# We replace it with a wrapper around standard LogisticRegression that accepts
# the same arguments but ignores CV-specific ones.


class MockLogisticRegressionCV(LogisticRegression):
    def __init__(self, Cs=10, cv=5, **kwargs):
        # Filter out arguments that LogisticRegression doesn't accept
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "n_jobs"}
        # Note: n_jobs is valid for LogisticRegression but we clean just in case of conflict
        # actually LogisticRegression accepts n_jobs.
        # We just need to remove Cs and cv.
        super().__init__(**kwargs)
        self.Cs = Cs
        self.cv = cv


print("Patching LogisticRegressionCV with Mock version for small dataset...")
library.training.LogisticRegressionCV = MockLogisticRegressionCV

# =============================================================================
# 5. Execute Pipeline
# =============================================================================
print("\n=== Starting Pipeline Execution ===")

# Verify Dataset Loading Logic
print("Verifying Dataset...")
ds, _ = library.dataset.get_dataloaders(
    library.config.STREAMS["stream_a"], batch_size=4
)
batch = next(iter(ds))
views, targets, ids = batch
assert "global" in views
assert "standard" in views
assert "local" in views
assert views["global"].shape == (4, 3, 224, 224)  # Batch 4, RGB, 224x224
print("Dataset verification passed.")

# Run the Ensemble Pipeline
# This will:
# 1. Extract features for Stream A (ConvNeXt) using the subset
# 2. Train the Mock Classifier for Stream A
# 3. Extract features for Stream B (MaxViT) using the subset
# 4. Train the Mock Classifier for Stream B
# 5. Optimize ensemble weights
# 6. Generate submission
library.ensemble.run_ensemble(load_cached_model=False)

# =============================================================================
# 6. Validate Results
# =============================================================================
print("\n=== Validating Submission ===")

if not os.path.exists(DEMO_SUBMISSION_PATH):
    raise FileNotFoundError(f"Submission file not found at {DEMO_SUBMISSION_PATH}")

df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)
print(f"Submission loaded. Shape: {df_sub.shape}")

# Check rows (should match test subset size)
expected_rows = len(test_subset)
if len(df_sub) != expected_rows:
    raise AssertionError(f"Expected {expected_rows} rows, got {len(df_sub)}")

# Check columns (id + 120 breeds)
expected_cols = 121
if df_sub.shape[1] != expected_cols:
    raise AssertionError(f"Expected {expected_cols} columns, got {df_sub.shape[1]}")

# Check ID column
if "id" not in df_sub.columns:
    raise AssertionError("Column 'id' missing from submission.")

# Check probability range
probs = df_sub.drop(columns=["id"]).values
if probs.min() < 0 or probs.max() > 1.0 + 1e-6:
    raise AssertionError("Probabilities out of range [0, 1]")

print("All validation checks passed successfully.")
print(f"Demo completed. Artifacts stored in {DEMO_WORKING_DIR}")
