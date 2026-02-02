import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Ensure the library path is accessible (current directory is root)
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import (
    WORKING_DIR,
    METADATA_DIR,
    INPUT_DIM,
    NUM_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    SEED,
)
from library.data_loader import load_and_process_data, GestureDataset, get_dataloaders
from library.model import ASH_KN, predict_sequence
from library.trainer import run_training_session
from library.inference import generate_submission
from library.utils import (
    levenshtein_distance,
    run_length_encoding,
    compute_levenshtein_score,
)


# ==========================================
# 0. Setup & Configuration
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(SEED)

# Define paths for mini metadata
MINI_TRAIN_PATH = os.path.join(WORKING_DIR, "mini_train.csv")
MINI_VAL_PATH = os.path.join(WORKING_DIR, "mini_val.csv")
MINI_TEST_PATH = os.path.join(WORKING_DIR, "mini_test.csv")

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

print("=== 1. Creating Mini-Metadata for Speed ===")
# Load original metadata
orig_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
orig_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
orig_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

# Sample top 5 rows for speed
mini_train = orig_train.head(5)
mini_val = orig_val.head(5)
mini_test = orig_test.head(5)

# Save to working directory
mini_train.to_csv(MINI_TRAIN_PATH, index=False)
mini_val.to_csv(MINI_VAL_PATH, index=False)
mini_test.to_csv(MINI_TEST_PATH, index=False)

print(
    f"Created mini datasets: Train={len(mini_train)}, Val={len(mini_val)}, Test={len(mini_test)}"
)

# ==========================================
# 1. Data Loading Demonstration
# ==========================================
print("\n=== 2. Demonstrating Data Loading ===")

# Use load_and_process_data with a unique cache name to avoid loading full dataset cache
# This function loads raw data, computes MFCC, parses skeletons, and aligns labels
train_data_dict = load_and_process_data(
    metadata_path=MINI_TRAIN_PATH,
    cache_name="dataset_mini_train",
    load_cached_data=False,  # Force re-compute for demo purposes
)

# Verify dictionary structure
sample_ids = list(train_data_dict.keys())
print(f"Loaded {len(sample_ids)} samples.")
first_sample = train_data_dict[sample_ids[0]]

assert "skeleton" in first_sample
assert "audio" in first_sample
assert "labels" in first_sample
print(f"Sample '{sample_ids[0]}' shapes:")
print(f"  Skeleton: {first_sample['skeleton'].shape}")
print(f"  Audio:    {first_sample['audio'].shape}")
print(f"  Labels:   {first_sample['labels'].shape}")

# Instantiate Dataset
# augment=True enables KinematicAugmentor (rotation/scaling)
train_dataset = GestureDataset(
    train_data_dict, augment=True, window_size=WINDOW_SIZE, stride=STRIDE
)
print(f"Dataset created with {len(train_dataset)} windows.")

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Fetch one batch
features, targets = next(iter(train_loader))

# Check shapes
# Features: (Batch, Window, InputDim)
# Targets: (Batch, Window)
print(f"Batch Features Shape: {features.shape}")
print(f"Batch Targets Shape: {targets.shape}")

assert features.shape[1] == WINDOW_SIZE
assert features.shape[2] == INPUT_DIM
assert targets.shape[1] == WINDOW_SIZE

# ==========================================
# 2. Model Demonstration
# ==========================================
print("\n=== 3. Demonstrating Model Architecture ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASH_KN().to(device)

# Move batch to device
inputs = features.to(device)
targets = targets.to(device)

# Forward Pass
# Model returns 3 outputs (logits from Stage 1, 2, and 3)
l1, l2, l3 = model(inputs)

print(f"Logits1 Shape: {l1.shape}")
print(f"Logits2 Shape: {l2.shape}")
print(f"Logits3 Shape: {l3.shape}")

# Validation
assert l3.shape == (inputs.shape[0], WINDOW_SIZE, NUM_CLASSES)
print("Model forward pass successful.")

# ==========================================
# 3. Utility Function Verification
# ==========================================
print("\n=== 4. Verifying Utilities ===")

# Test Levenshtein
seq1 = [1, 2, 3]
seq2 = [1, 2, 4]
dist = levenshtein_distance(seq1, seq2)
print(f"Levenshtein([1,2,3], [1,2,4]) = {dist}")
assert dist == 1

# Test Run-Length Encoding
# Input: [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2] (assuming min_duration=5)
# 0 is background, should be ignored
raw_preds = np.array([1] * 5 + [0] * 5 + [2] * 5)
decoded = run_length_encoding(raw_preds, min_duration=5)
print(f"RLE Input: {raw_preds}")
print(f"RLE Output: {decoded}")
assert decoded == [1, 2]

# ==========================================
# 4. Training Pipeline Demonstration
# ==========================================
print("\n=== 5. Running Short Training Session ===")

# We need to monkey-patch the metadata paths in library.config or library.trainer?
# The library code imports constants directly. We cannot easily change them inside the library.
# However, run_training_session calls get_dataloaders which calls load_and_process_data with FIXED paths.
# To make this work with our mini-datasets without modifying library code,
# we must manually invoke the training loop components or rely on the fact that
# we can overwrite the cache files the library expects, OR we accept that we can't change the path constant.

# Strategy:
# Since we cannot modify library files to point to MINI_TRAIN_PATH, we will manually
# construct the loaders using our mini data and call train_one_epoch/validate loop manually
# to demonstrate the logic, instead of calling run_training_session directly which is hardcoded.

# 1. Load Mini Val Data
val_data_dict = load_and_process_data(
    metadata_path=MINI_VAL_PATH, cache_name="dataset_mini_val", load_cached_data=False
)
val_dataset = GestureDataset(val_data_dict, augment=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

# 2. Setup Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Manual Loop (2 Epochs)
print("Starting manual training loop on mini-dataset...")
for epoch in range(2):
    # Train step (using the train_loader we created in Step 1)
    from library.model import train_one_epoch, validate

    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# Save this "trained" model
demo_model_path = os.path.join(WORKING_DIR, "demo_model.pth")
torch.save(model.state_dict(), demo_model_path)
print(f"Saved demo model to {demo_model_path}")

# ==========================================
# 5. Inference & Submission Demonstration
# ==========================================
print("\n=== 6. Generating Submission ===")

# We will use the generate_submission logic but adapted for our mini-test set
# Again, generate_submission in library.inference uses get_test_loader which uses TEST_METADATA_PATH.
# We will manually load mini test data and run inference.

# Load Mini Test Data
test_data_dict = load_and_process_data(
    metadata_path=MINI_TEST_PATH, cache_name="dataset_mini_test", load_cached_data=False
)

results = []
model.eval()

print("Predicting on mini test set...")
for sid in sorted(test_data_dict.keys()):
    sample = test_data_dict[sid]
    skel = sample["skeleton"]
    audio = sample["audio"]

    if skel is not None and len(skel) > 0:
        # Use predict_sequence from library.model
        # This handles sliding window inference internally
        preds = predict_sequence(model, skel, audio, device)

        # Decode
        pred_seq = run_length_encoding(preds, min_duration=5)
        pred_str = ",".join(map(str, pred_seq))
    else:
        pred_str = ""

    results.append(f"{sid},{pred_str}")
    print(f"  {sid} -> [{pred_str}]")

# Save submission
submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")
with open(submission_path, "w") as f:
    for line in results:
        f.write(line + "\n")

print(f"Submission saved to {submission_path}")
print("\n=== Demonstration Complete ===")
