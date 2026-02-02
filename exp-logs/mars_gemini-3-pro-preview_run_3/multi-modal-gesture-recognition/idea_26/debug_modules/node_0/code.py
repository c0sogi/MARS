import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import provided library components
from library.config import Config
from library.utils import levenshtein_distance, compute_levenshtein_score
from library.data_loader import get_data_loaders
from library.model import HCNCSN
from library.trainer import Trainer, set_seed


def create_subset_metadata(source_csv, dest_csv, num_samples=10):
    """
    Reads the first 'num_samples' from the source CSV and saves to dest_csv.
    This allows us to run the pipeline quickly on a small dataset.
    """
    if not os.path.exists(source_csv):
        print(f"Warning: Source CSV {source_csv} not found. Creating dummy.")
        return

    df = pd.read_csv(source_csv)
    # Take a subset
    subset = df.head(num_samples)
    subset.to_csv(dest_csv, index=False)
    print(f"Created subset metadata at {dest_csv} with {len(subset)} samples.")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("=== Setting up Demo Environment ===")

    # Define demo paths
    DEMO_DIR = "./working/demo_run"
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Clean previous run
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Patch the Config class to use demo paths and reduced hyperparameters
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "mstcn_demo_model.pth")

    # Speed up training for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.EARLY_STOPPING_PATIENCE = 2

    # Create subset CSVs
    subset_train_path = os.path.join(DEMO_DIR, "train_subset.csv")
    subset_val_path = os.path.join(DEMO_DIR, "val_subset.csv")
    subset_test_path = os.path.join(DEMO_DIR, "test_subset.csv")

    create_subset_metadata(
        os.path.join("./metadata", "train.csv"), subset_train_path, num_samples=10
    )
    create_subset_metadata(
        os.path.join("./metadata", "val.csv"), subset_val_path, num_samples=10
    )
    create_subset_metadata(
        os.path.join("./metadata", "test.csv"), subset_test_path, num_samples=10
    )

    # Point Config to subsets
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    Config.TEST_CSV = subset_test_path

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Utility Verification
    # ==========================================
    print("\n=== Verifying Utilities ===")
    # Test Levenshtein distance
    dist = levenshtein_distance([1, 2, 3], [1, 2, 3])
    assert dist == 0, f"Expected distance 0, got {dist}"

    dist = levenshtein_distance([1, 2, 3], [1, 3])
    assert dist == 1, f"Expected distance 1 (deletion), got {dist}"

    dist = levenshtein_distance([1, 2], [3, 4])
    assert dist == 2, f"Expected distance 2 (substitutions), got {dist}"
    print("Levenshtein distance logic verified.")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("\n=== Loading Data (Subset) ===")
    # load_cached_data=False forces processing from raw files to verify loader logic
    train_loader, val_loader, val_metric_loader, test_loader = get_data_loaders(
        load_cached_data=False
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Verify Batch Shape
    # Train loader returns (features, labels)
    features, labels = next(iter(train_loader))
    print(
        f"Batch Features Shape: {features.shape}"
    )  # Expected: (Batch, Window, InputDim)
    print(f"Batch Labels Shape: {labels.shape}")  # Expected: (Batch, Window)

    assert features.shape[0] == Config.BATCH_SIZE
    assert features.shape[1] == Config.WINDOW_SIZE
    assert features.shape[2] == Config.INPUT_DIM
    assert labels.shape[0] == Config.BATCH_SIZE
    assert labels.shape[1] == Config.WINDOW_SIZE
    print("Data Loader shapes verified.")

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    print("\n=== Verifying Model Architecture ===")
    model = HCNCSN()
    # Create dummy input: (Batch, Time, InputDim)
    dummy_input = torch.randn(2, 64, Config.INPUT_DIM)

    # Forward pass
    l1, l2, l3 = model(dummy_input)

    print(f"Logits 1 Shape: {l1.shape}")
    print(f"Logits 2 Shape: {l2.shape}")
    print(f"Logits 3 Shape: {l3.shape}")

    assert l1.shape == (2, 64, Config.NUM_CLASSES)
    assert l2.shape == (2, 64, Config.NUM_CLASSES)
    assert l3.shape == (2, 64, Config.NUM_CLASSES)
    print("Model forward pass verified.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n=== Starting Training Demo ===")
    trainer = Trainer()

    # Note: trainer.fit expects a validation loader that returns full sequences (3 values)
    # because it calls self.validate() which unpacks 3 items.
    # val_metric_loader provides (features, targets, sid).
    trainer.fit(train_loader, val_metric_loader)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created."
    print("Training demo completed and model saved.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n=== Generating Submission ===")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    trainer.generate_submission(test_loader, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Check submission content
    with open(submission_path, "r") as f:
        lines = f.readlines()
        print(f"Generated {len(lines)} prediction lines.")
        if len(lines) > 0:
            print(f"Sample prediction: {lines[0].strip()}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
