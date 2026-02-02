import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
import time

# Import library modules
# We import them to access their namespaces for patching
import library.config as config
import library.data_utils as data_utils
import library.extract_features as extract_features
import library.feature_dataset as feature_dataset
import library.cascade_model as cascade_model
import library.trainer as trainer

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
WORKING_DIR = "./working"
DEMO_DIR = os.path.join(WORKING_DIR, "demo_execution")
INPUT_DIR = "./input"

# Ensure clean state
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"=== Starting Demonstration in {DEMO_DIR} ===")

# ==========================================
# 1. PREPARE MINI DATASET (METADATA)
# ==========================================
print("\n[1] Preparing Mini Metadata...")

# Load original metadata
orig_train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
orig_val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))
orig_test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

# Sample first 50 records for speed
mini_train_meta = orig_train_meta.head(50)
mini_val_meta = orig_val_meta.head(50)
mini_test_meta = orig_test_meta.head(50)

# Save mini metadata to demo directory
demo_train_meta_path = os.path.join(DEMO_DIR, "train.csv")
demo_val_meta_path = os.path.join(DEMO_DIR, "val.csv")
demo_test_meta_path = os.path.join(DEMO_DIR, "test.csv")

mini_train_meta.to_csv(demo_train_meta_path, index=False)
mini_val_meta.to_csv(demo_val_meta_path, index=False)
mini_test_meta.to_csv(demo_test_meta_path, index=False)

print(
    f"Saved mini metadata: Train={len(mini_train_meta)}, Val={len(mini_val_meta)}, Test={len(mini_test_meta)}"
)

# Define paths for extracted features
demo_train_feats = os.path.join(DEMO_DIR, "train_features.npy")
demo_train_labels = os.path.join(DEMO_DIR, "train_labels.npy")
demo_val_feats = os.path.join(DEMO_DIR, "val_features.npy")
demo_val_labels = os.path.join(DEMO_DIR, "val_labels.npy")
demo_test_feats = os.path.join(DEMO_DIR, "test_features.npy")
demo_test_ids = os.path.join(DEMO_DIR, "test_ids.npy")
demo_submission = os.path.join(DEMO_DIR, "submission.csv")
demo_model_path = os.path.join(DEMO_DIR, "demo_model.pth")

# ==========================================
# 2. PATCH LIBRARY MODULES
# ==========================================
print("\n[2] Patching Library Configuration for Demo...")

# Patch extract_features module to use our mini metadata and output paths
extract_features.TRAIN_META_PATH = demo_train_meta_path
extract_features.VAL_META_PATH = demo_val_meta_path
extract_features.TEST_META_PATH = demo_test_meta_path

extract_features.TRAIN_FEATURES_PATH = demo_train_feats
extract_features.TRAIN_LABELS_PATH = demo_train_labels
extract_features.VAL_FEATURES_PATH = demo_val_feats
extract_features.VAL_LABELS_PATH = demo_val_labels
extract_features.TEST_FEATURES_PATH = demo_test_feats
extract_features.TEST_IDS_PATH = demo_test_ids
extract_features.CACHE_DIR = DEMO_DIR

# Disable multiprocessing for small demo to avoid overhead
extract_features.NUM_WORKERS = 0
trainer.NUM_WORKERS = 0

# Patch data_utils to cache hierarchy map in demo dir
data_utils.CACHE_DIR = DEMO_DIR

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
print("\n[3] Running Feature Extraction (ResNet50 + EfficientNet-B0)...")
# This will read images from the actual BSON files using the offsets in our mini metadata
# and run them through the backbone models.
start_time = time.time()
extract_features.extract_and_save(load_cached_data=False)
print(f"Extraction finished in {time.time() - start_time:.2f}s")

# Verify outputs
assert os.path.exists(demo_train_feats), "Train features not saved"
assert os.path.exists(demo_train_labels), "Train labels not saved"
feats = np.load(demo_train_feats)
print(f"Feature shape verification: {feats.shape}")
assert feats.shape == (50, 3328), f"Expected (50, 3328), got {feats.shape}"

# ==========================================
# 4. HIERARCHY MANAGER VERIFICATION
# ==========================================
print("\n[4] Verifying Hierarchy Manager...")
hm = data_utils.HierarchyManager(load_cached_data=False)
assert os.path.exists(
    os.path.join(DEMO_DIR, "hierarchy_map.parquet")
), "Hierarchy map not cached"

# Test mapping logic
sample_cat_id = mini_train_meta.iloc[0]["category_id"]
l1, l2, l3 = hm.get_labels(sample_cat_id)
print(f"Mapped Category ID {sample_cat_id} -> L1:{l1}, L2:{l2}, L3:{l3}")
assert isinstance(l1, (int, np.integer))
assert isinstance(l2, (int, np.integer))
assert isinstance(l3, (int, np.integer))

# ==========================================
# 5. DATALOADER & COLLATOR VERIFICATION
# ==========================================
print("\n[5] Verifying DataLoaders and MixUp...")
train_loader, val_loader, test_loader = feature_dataset.get_dataloaders(
    train_features_path=demo_train_feats,
    train_labels_path=demo_train_labels,
    val_features_path=demo_val_feats,
    val_labels_path=demo_val_labels,
    test_features_path=demo_test_feats,
    test_ids_path=demo_test_ids,
    hierarchy_manager=hm,
    batch_size=16,
    mixup_alpha=0.2,
    num_workers=0,
)

# Fetch one batch from train loader
features, targets_a, targets_b, lam = next(iter(train_loader))
print(f"Batch shapes: Features={features.shape}, Lambda={lam}")
assert features.shape == (16, 3328)
assert len(targets_a) == 3  # (l1, l2, l3)
assert isinstance(lam, float)

# ==========================================
# 6. MODEL ARCHITECTURE VERIFICATION
# ==========================================
print("\n[6] Verifying Cascade Model Architecture...")
model = cascade_model.ConditionalCascadeMLP()
dummy_input = torch.randn(4, 3328)
l1_logits, l2_logits, l3_logits = model(dummy_input)

print(f"Logit shapes: L1={l1_logits.shape}, L2={l2_logits.shape}, L3={l3_logits.shape}")
assert l1_logits.shape == (4, 49)  # 49 L1 classes
assert l2_logits.shape == (4, 483)  # 483 L2 classes
assert l3_logits.shape == (4, 5270)  # 5270 L3 classes

# ==========================================
# 7. TRAINING PIPELINE EXECUTION
# ==========================================
print("\n[7] Running Full Training Pipeline...")

# Initialize Trainer with our demo paths
# We manually instantiate Trainer to override the save path
demo_trainer = trainer.Trainer(hierarchy_manager=hm, model_save_path=demo_model_path)

# Run Fit
print("Training for 2 epochs...")
demo_trainer.fit(train_loader, val_loader, epochs=2, patience=2)

# Verify model saved
assert os.path.exists(demo_model_path), "Model checkpoint not saved"

# Run Predict
print("Generating predictions...")
demo_trainer.predict(test_loader, output_csv_path=demo_submission)

# Verify submission
assert os.path.exists(demo_submission), "Submission file not created"
df_sub = pd.read_csv(demo_submission)
print("Submission Head:")
print(df_sub.head())

assert len(df_sub) == 50, f"Expected 50 predictions, got {len(df_sub)}"
assert "_id" in df_sub.columns and "category_id" in df_sub.columns

print("\n=== Demonstration Completed Successfully ===")
