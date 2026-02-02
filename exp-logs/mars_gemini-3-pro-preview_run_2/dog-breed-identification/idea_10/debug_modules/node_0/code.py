import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import shutil
from sklearn.linear_model import LogisticRegressionCV

# ==========================================
# 1. Configuration & Patching
# ==========================================
from library.config import Config

# Define a separate working directory for this demo to avoid conflicts
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

print(f"--- Configuration ---")
print(f"Working Directory: {DEMO_DIR}")

# Patch Config to use the demo directory
Config.WORK_DIR = DEMO_DIR
Config.SUBMISSION_DIR = DEMO_DIR
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

# Patch Cache Paths
Config.CACHE_A_TRAIN_EMB = os.path.join(DEMO_DIR, "stream_a_train_emb.npy")
Config.CACHE_A_VAL_EMB = os.path.join(DEMO_DIR, "stream_a_val_emb.npy")
Config.CACHE_A_TEST_EMB = os.path.join(DEMO_DIR, "stream_a_test_emb.npy")
Config.CACHE_B_TRAIN_EMB = os.path.join(DEMO_DIR, "stream_b_train_emb.npy")
Config.CACHE_B_VAL_EMB = os.path.join(DEMO_DIR, "stream_b_val_emb.npy")
Config.CACHE_B_TEST_EMB = os.path.join(DEMO_DIR, "stream_b_test_emb.npy")

Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels.npy")
Config.CACHE_VAL_LABELS = os.path.join(DEMO_DIR, "val_labels.npy")
Config.CACHE_TRAIN_IDS = os.path.join(DEMO_DIR, "train_ids.npy")
Config.CACHE_VAL_IDS = os.path.join(DEMO_DIR, "val_ids.npy")
Config.CACHE_TEST_IDS = os.path.join(DEMO_DIR, "test_ids.npy")

Config.MODEL_A_HEAD_PATH = os.path.join(DEMO_DIR, "model_a.joblib")
Config.MODEL_B_HEAD_PATH = os.path.join(DEMO_DIR, "model_b.joblib")
Config.ENSEMBLE_WEIGHTS_PATH = os.path.join(DEMO_DIR, "weights.json")

# Patch Hyperparameters for Speed
Config.BATCH_SIZE = 8
Config.CV_FOLDS = 2  # Minimum for CV
Config.CS_COUNT = 1  # Only check 1 value for Regularization Strength
Config.MAX_ITER = 50  # Limit iterations for convergence
Config.DEBUG_SAMPLE_SIZE = None  # We control size via metadata files

# Set Reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. Data Preparation (Subset Creation)
# ==========================================
print("\n--- Preparing Data Subset ---")
# We create custom metadata files with only 3 classes and a few samples
# to ensure LogisticRegressionCV runs fast and has enough samples per class.

original_train = pd.read_csv("./metadata/train.csv")
original_val = pd.read_csv("./metadata/val.csv")
original_test = pd.read_csv("./metadata/test.csv")

# Select top 3 breeds
selected_breeds = original_train["breed"].unique()[:3]
print(f"Selected Breeds for Demo: {selected_breeds}")

# Filter Train (5 samples per breed)
demo_train = (
    original_train[original_train["breed"].isin(selected_breeds)]
    .groupby("breed")
    .head(5)
    .reset_index(drop=True)
)
# Filter Val (3 samples per breed)
demo_val = (
    original_val[original_val["breed"].isin(selected_breeds)]
    .groupby("breed")
    .head(3)
    .reset_index(drop=True)
)
# Filter Test (Just take first 10 images)
demo_test = original_test.head(10).reset_index(drop=True)

# Save to demo directory
demo_train_path = os.path.join(DEMO_DIR, "train.csv")
demo_val_path = os.path.join(DEMO_DIR, "val.csv")
demo_test_path = os.path.join(DEMO_DIR, "test.csv")

demo_train.to_csv(demo_train_path, index=False)
demo_val.to_csv(demo_val_path, index=False)
demo_test.to_csv(demo_test_path, index=False)

# Point Config to these new files
Config.TRAIN_METADATA_PATH = demo_train_path
Config.VAL_METADATA_PATH = demo_val_path
Config.TEST_METADATA_PATH = demo_test_path

print(
    f"Created subset metadata: Train={len(demo_train)}, Val={len(demo_val)}, Test={len(demo_test)}"
)

# ==========================================
# 3. Import Library Modules
# ==========================================
# Importing after config patching ensures consistency if modules read config at import time
from library.dataset import DogDataset, get_transforms
from library.model_factory import get_backbone
from library.classifier import StreamClassifier

# ==========================================
# 4. Demonstration & Verification
# ==========================================


def demo_dataset():
    print("\n[1/3] Verifying DogDataset...")
    transform = get_transforms("stream_a", "standard")
    ds = DogDataset(Config.TRAIN_METADATA_PATH, transform=transform)

    # Assertions
    assert len(ds) == 15, f"Expected 15 samples, got {len(ds)}"
    assert ds.get_num_classes() == 3, f"Expected 3 classes, got {ds.get_num_classes()}"

    img, label, img_id = ds[0]
    # Standard view is 224x224
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, int), "Label should be an integer"
    print("Dataset verification passed.")


def demo_model_backbone():
    print("\n[2/3] Verifying Model Backbone (ConvNeXt)...")
    # Load Stream A backbone
    model = get_backbone("stream_a")
    model = model.to(Config.DEVICE)

    # Check if parameters are frozen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (
        trainable_params == 0
    ), "Backbone parameters should be frozen (requires_grad=False)"

    # Check Output Shape (Forward Pass)
    dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    # ConvNeXt Large embedding dimension is 1536
    # Output shape should be (Batch, Features)
    assert output.shape == (2, 1536), f"Expected shape (2, 1536), got {output.shape}"
    print("Model backbone verification passed.")

    # Cleanup
    del model
    torch.cuda.empty_cache()


def demo_full_pipeline():
    print("\n[3/3] Running Full Pipeline (StreamClassifier)...")

    classifier = StreamClassifier()

    # Run the full pipeline: Feature Extraction -> Training -> Optimization -> Submission
    # This uses the reduced dataset and hyperparameters defined in Config
    classifier.run()

    # Verify Outputs
    print("Verifying pipeline outputs...")

    # 1. Check Model Files
    assert os.path.exists(Config.MODEL_A_HEAD_PATH), "Model A file missing"
    assert os.path.exists(Config.MODEL_B_HEAD_PATH), "Model B file missing"
    assert os.path.exists(Config.ENSEMBLE_WEIGHTS_PATH), "Ensemble weights file missing"

    # 2. Check Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    # Rows should equal test set size (10)
    # Columns should be id + 3 classes
    assert len(df_sub) == 10, f"Expected 10 predictions, got {len(df_sub)}"
    assert (
        len(df_sub.columns) == 4
    ), f"Expected 4 columns (id + 3 breeds), got {len(df_sub.columns)}"
    assert "id" in df_sub.columns, "Column 'id' missing"

    # Check probabilities sum to 1
    prob_cols = [c for c in df_sub.columns if c != "id"]
    row_sums = df_sub[prob_cols].sum(axis=1)
    # Allow small float error
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Pipeline verification passed.")
    print(f"Submission generated at: {Config.SUBMISSION_PATH}")
    print("Sample Submission:\n", df_sub.head(3))


if __name__ == "__main__":
    try:
        demo_dataset()
        demo_model_backbone()
        demo_full_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        # Re-raise to ensure the task is marked as failed if something goes wrong
        raise e
