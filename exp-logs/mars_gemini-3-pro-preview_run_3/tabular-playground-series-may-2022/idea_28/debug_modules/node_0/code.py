import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path to import library modules correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import get_dataloaders
from library.model_utils import IPPFEModel
from library.train_utils import train, generate_submission


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("Starting Demo Execution...")

    # 1. Configuration Setup for Rapid Demo
    # We modify the Config class attributes directly to create a fast, isolated run.
    print("Configuring environment...")

    # Create a separate working directory for this demo
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    Config.MODEL_PATH = os.path.join(demo_working_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Reduce dataset size and training duration for speed
    Config.MAX_SAMPLES = 2048  # Small subset for demonstration
    Config.BATCH_SIZE = 256
    Config.EPOCHS = 2  # Minimal epochs to prove the loop works
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading & Verification
    print("\nInitializing DataLoaders...")
    # load_cached_data=False forces reprocessing to ensure we test the feature engineering logic
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        load_cached_data=False, verbose=True
    )

    # Assertions to verify data loading
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert "vocab_sizes" in metadata, "Metadata missing vocab_sizes."
    assert "num_cont_features" in metadata, "Metadata missing num_cont_features."

    # Inspect a single batch
    batch = next(iter(train_loader))
    cat_features = batch["cat_features"]
    cont_features = batch["cont_features"]
    targets = batch["target"]

    print(
        f"Batch Shapes -> Cat: {cat_features.shape}, Cont: {cont_features.shape}, Target: {targets.shape}"
    )

    # Verify feature dimensions
    # Cat features: 10 (f_27 decomposed) + 1 (f_29) + 1 (f_30) = 12
    assert (
        cat_features.shape[1] == 12
    ), f"Expected 12 categorical features, got {cat_features.shape[1]}"
    # Continuous features: 32 total - 1 (id) - 1 (target) - 1 (source) - 1 (f_27) - 2 (f_29, f_30) + 1 (unique_char_count) = 27?
    # Let's rely on metadata for the exact count, but ensure consistency
    assert (
        cont_features.shape[1] == metadata["num_cont_features"]
    ), "Continuous feature count mismatch."

    # 3. Model Initialization & Verification
    print("\nInitializing Model...")
    model = IPPFEModel(
        vocab_sizes=metadata["vocab_sizes"], num_cont=metadata["num_cont_features"]
    ).to(Config.DEVICE)

    # Perform a dummy forward pass to verify architecture
    model.eval()
    with torch.no_grad():
        cat_x = cat_features.to(Config.DEVICE)
        cont_x = cont_features.to(Config.DEVICE)
        logits = model(cat_x, cont_x)

    print(f"Model Output Shape: {logits.shape}")
    # Expecting (Batch_Size, 5) because IPPFE has 5 streams
    assert logits.shape == (
        cat_features.shape[0],
        5,
    ), f"Expected output shape (B, 5), got {logits.shape}"

    # 4. Training Loop
    print("\nStarting Training...")
    best_auc = train(train_loader, val_loader, metadata)

    print(f"Training finished. Best AUC: {best_auc:.4f}")

    # Verify model artifact creation
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    assert os.path.getsize(Config.MODEL_PATH) > 0, "Model file is empty."

    # 5. Inference & Submission
    print("\nGenerating Submission...")
    # Note: test_loader uses the full test set (100k rows) as per data_utils logic (MAX_SAMPLES only applies to train/val)
    # This is fine, inference is fast.
    generate_submission(test_loader, metadata)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Check format
    assert list(df_sub.columns) == ["id", "target"], "Submission columns are incorrect."
    assert (
        df_sub.shape[0] == 100000
    ), f"Expected 100,000 predictions, got {df_sub.shape[0]}"

    # Check value range
    preds = df_sub["target"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions are out of probability range [0, 1]."

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
