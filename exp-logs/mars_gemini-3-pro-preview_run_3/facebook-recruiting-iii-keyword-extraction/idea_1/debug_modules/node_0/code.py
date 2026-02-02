import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# 1. Configuration Patching
# =============================================================================
# We import config first and modify it to use a small subset of data and
# reduced hyperparameters for a fast demonstration.
import library.config as config

print("Setting up demonstration configuration...")

# Define a separate working directory for this demo to avoid conflicts
DEMO_DIR = "./working/demo_pipeline"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Patch Paths
config.WORKING_DIR = DEMO_DIR
config.SUBMISSION_DIR = DEMO_DIR
config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

# Patch Cache Paths (Point to demo directory so we don't use/overwrite real full-data caches)
config.TFIDF_VECTORIZER_PATH = os.path.join(DEMO_DIR, "tfidf.joblib")
config.MLB_PATH = os.path.join(DEMO_DIR, "mlb.joblib")
config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.npz")
config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.npz")
config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.npz")
config.TRAIN_LABELS_PATH = os.path.join(DEMO_DIR, "train_labels.npz")
config.VAL_LABELS_PATH = os.path.join(DEMO_DIR, "val_labels.npz")
config.MODEL_PATH = os.path.join(DEMO_DIR, "model.pth")

# Patch Metadata Paths (We will create these subset files shortly)
config.TRAIN_META_PATH = os.path.join(DEMO_DIR, "train_meta.csv")
config.VAL_META_PATH = os.path.join(DEMO_DIR, "val_meta.csv")
config.TEST_META_PATH = os.path.join(DEMO_DIR, "test_meta.csv")

# Patch Hyperparameters for Speed
config.MAX_FEATURES = 1000  # Reduced vocabulary size
config.TOP_K_TAGS = 50  # Reduced output space
config.EPOCHS = 1  # Single epoch
config.BATCH_SIZE = 128  # Reasonable batch size
config.HIDDEN_DIM = 64  # Smaller model dimension
config.NUM_WORKERS = 2  # Reduce overhead

# =============================================================================
# 2. Data Subsetting
# =============================================================================
print("Creating data subsets for demonstration...")


def create_subset(src_path, dst_path, n=1000):
    """Reads the first n rows of the source metadata and saves to dst_path."""
    if os.path.exists(src_path):
        df = pd.read_csv(src_path).head(n)
        df.to_csv(dst_path, index=False)
        print(f"Created subset: {dst_path} ({len(df)} rows)")
    else:
        raise FileNotFoundError(f"Source metadata not found: {src_path}")


# Original paths
ORIG_TRAIN_META = "./metadata/train.csv"
ORIG_VAL_META = "./metadata/val.csv"
ORIG_TEST_META = "./metadata/test.csv"

# Create subsets
create_subset(ORIG_TRAIN_META, config.TRAIN_META_PATH, n=2000)
create_subset(ORIG_VAL_META, config.VAL_META_PATH, n=500)
create_subset(ORIG_TEST_META, config.TEST_META_PATH, n=500)

# =============================================================================
# 3. Import Library Modules
# =============================================================================
# Importing these AFTER patching config ensures they use the modified values
from library.data_processor import process_data
from library.dataset import get_dataloaders
from library.model import SparseMLP
from library.trainer import run_training, generate_submission
from library.utils import calculate_f1_score

# =============================================================================
# 4. Main Execution Flow
# =============================================================================
if __name__ == "__main__":
    config.set_seed()

    # --- Step A: Verify Metric Logic ---
    print("\n--- Step A: Verifying Metric Logic ---")
    # Test F1 calculation with known values
    y_true = ["python java", "c++", "php"]
    y_pred = ["python", "c++ java", "php"]
    # Case 1: 1/1 precision, 1/2 recall -> F1 ~0.67
    # Case 2: 1/2 precision, 1/1 recall -> F1 ~0.67
    # Case 3: Perfect match -> F1 1.0
    score = calculate_f1_score(y_true, y_pred)
    print(f"Calculated F1: {score:.4f}")
    assert 0.7 < score < 0.8, "F1 Score calculation logic seems incorrect."

    # --- Step B: Data Processing ---
    print("\n--- Step B: Running Data Processing ---")
    # Force processing from scratch (load_cached_data=False) to test the pipeline
    X_train, y_train, X_val, y_val, X_test, fe = process_data(load_cached_data=False)

    # Validations
    print(f"Train Features Shape: {X_train.shape}")
    print(f"Train Labels Shape: {y_train.shape}")

    # Check Feature Dimensions match our patched config
    assert (
        X_train.shape[1] == config.MAX_FEATURES
    ), f"Expected {config.MAX_FEATURES} features, got {X_train.shape[1]}"

    # Check Label Dimensions match our patched config
    assert (
        y_train.shape[1] == config.TOP_K_TAGS
    ), f"Expected {config.TOP_K_TAGS} labels, got {y_train.shape[1]}"

    # Check Feature Engineer State
    assert len(fe.mlb.classes_) == config.TOP_K_TAGS
    assert len(fe.tfidf.vocabulary_) <= config.MAX_FEATURES

    # --- Step C: Dataloaders ---
    print("\n--- Step C: Initializing Dataloaders ---")
    # We use get_dataloaders which internally calls process_data (will use cache this time)
    train_loader, val_loader, test_loader, fe_loaded = get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # Verify batch structure
    sample_x, sample_y = next(iter(train_loader))
    print(f"Batch X shape: {sample_x.shape}")
    print(f"Batch Y shape: {sample_y.shape}")

    assert sample_x.shape[1] == config.MAX_FEATURES
    assert sample_y.shape[1] == config.TOP_K_TAGS

    # --- Step D: Model Initialization ---
    print("\n--- Step D: Initializing Model ---")
    device = config.DEVICE
    model = SparseMLP(
        input_dim=config.MAX_FEATURES,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.TOP_K_TAGS,
    )
    model.to(device)

    # Forward pass check
    with torch.no_grad():
        logits = model(sample_x.to(device))
    assert logits.shape == (
        sample_x.shape[0],
        config.TOP_K_TAGS,
    ), "Model output shape mismatch"

    # --- Step E: Training ---
    print("\n--- Step E: Running Training Loop ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config.EPOCHS,
        patience=1,
        feature_engineer=fe_loaded,
    )

    # Verify model file created
    assert os.path.exists(
        config.MODEL_PATH
    ), f"Model file was not saved at {config.MODEL_PATH}"

    # --- Step F: Inference ---
    print("\n--- Step F: Running Inference ---")
    # We use the trainer's generate_submission function
    # Using a low threshold to ensure we get some tags for demonstration purposes
    preds = generate_submission(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        feature_engineer=fe_loaded,
        threshold=0.1,
        submission_file=config.SUBMISSION_FILE,
    )

    # Verify Submission
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file not created"

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print("Submission Head:")
    print(df_sub.head())

    # Check alignment with test metadata
    test_meta_len = len(pd.read_csv(config.TEST_META_PATH))
    assert (
        len(df_sub) == test_meta_len
    ), f"Submission row count ({len(df_sub)}) does not match test metadata ({test_meta_len})"

    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission columns incorrect"

    print("\n=== Demonstration Completed Successfully ===")
