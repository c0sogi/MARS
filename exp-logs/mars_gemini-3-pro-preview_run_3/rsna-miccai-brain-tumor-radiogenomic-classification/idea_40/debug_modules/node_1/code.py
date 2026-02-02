import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import provided library modules
from library.utils import process_patient, load_data_and_cache, set_seed
from library.data_loader import get_dataloaders
from library.model import SiameseNetwork
from library.train import run_training
from library.predict import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Demonstration Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Define directories
    INPUT_DIR = "./input"
    ORIGINAL_META_DIR = "./metadata"
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    print(f"Working directory: {WORKING_DIR}")

    # ==========================================
    # 2. Create Subset Metadata (Optimization)
    # ==========================================
    print("\n[Step 1] Creating subset metadata for rapid execution...")

    # Load original metadata
    train_df = pd.read_parquet(os.path.join(ORIGINAL_META_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(ORIGINAL_META_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(ORIGINAL_META_DIR, "test.parquet"))

    # Create subsets (e.g., 4 samples each to fit in one batch)
    subset_size = 4
    demo_train_df = train_df.head(subset_size).copy()
    demo_val_df = val_df.head(subset_size).copy()
    demo_test_df = test_df.head(subset_size).copy()

    # Save subset metadata
    demo_train_path = os.path.join(WORKING_DIR, "train.parquet")
    demo_val_path = os.path.join(WORKING_DIR, "val.parquet")
    demo_test_path = os.path.join(WORKING_DIR, "test.parquet")

    demo_train_df.to_parquet(demo_train_path, index=False)
    demo_val_df.to_parquet(demo_val_path, index=False)
    demo_test_df.to_parquet(demo_test_path, index=False)

    print(
        f"Subset metadata saved. Train: {len(demo_train_df)}, Val: {len(demo_val_df)}, Test: {len(demo_test_df)}"
    )

    # ==========================================
    # 3. Demonstrate Library: Utils
    # ==========================================
    print("\n[Step 2] Verifying library.utils...")

    # Test process_patient on a single row
    sample_row = demo_train_df.iloc[0]
    print(f"Processing patient: {sample_row['BraTS21ID']}...")

    # Note: process_patient returns (X_even, X_odd)
    # Expected shape: (Channels, Height, Width) -> (64, 224, 224)
    # 4 modalities * 16 slices per stream = 64 channels
    xe, xo = process_patient(sample_row, input_dir=INPUT_DIR)

    print(f"Processed shapes - Even: {xe.shape}, Odd: {xo.shape}")

    assert xe.shape == (64, 224, 224), f"Expected (64, 224, 224), got {xe.shape}"
    assert xo.shape == (64, 224, 224), f"Expected (64, 224, 224), got {xo.shape}"
    assert xe.dtype == np.float32, "Data type should be float32"

    # Test load_data_and_cache
    print("Testing load_data_and_cache...")
    X_e, X_o, y, ids = load_data_and_cache(
        metadata_path=demo_train_path,
        cache_dir=CACHE_DIR,
        load_cached_data=False,  # Force processing
        input_dir=INPUT_DIR,
        dataset_name="demo_train",
    )

    assert len(X_e) == subset_size
    assert os.path.exists(os.path.join(CACHE_DIR, "X_demo_train_even.npy"))
    print("Caching mechanism verified.")

    # ==========================================
    # 4. Demonstrate Library: Data Loader
    # ==========================================
    print("\n[Step 3] Verifying library.data_loader...")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_meta_path=demo_train_path,
        val_meta_path=demo_val_path,
        test_meta_path=demo_test_path,
        batch_size=2,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
        load_cached_data=True,
        cache_dir=CACHE_DIR,
    )

    # Fetch one batch
    batch_xe, batch_xo, batch_y = next(iter(train_loader))

    print(f"Batch shapes - Input: {batch_xe.shape}, Label: {batch_y.shape}")

    # Assertions
    assert batch_xe.shape == (2, 64, 224, 224)
    assert batch_y.shape == (2,)
    assert isinstance(batch_xe, torch.Tensor)
    print("DataLoaders verified.")

    # ==========================================
    # 5. Demonstrate Library: Model
    # ==========================================
    print("\n[Step 4] Verifying library.model...")

    model = SiameseNetwork(model_name="efficientnet_b0", pretrained=False)

    # Move to CPU for this quick demo check (or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_xe = batch_xe.to(device)
    batch_xo = batch_xo.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(batch_xe, batch_xo)

    print(f"Model output shape: {logits.shape}")

    assert logits.shape == (2, 1), f"Expected (2, 1), got {logits.shape}"
    print("Model architecture verified.")

    # ==========================================
    # 6. Demonstrate Library: Train
    # ==========================================
    print("\n[Step 5] Verifying library.train (Running 1 Epoch)...")

    # We use the run_training function which orchestrates the whole loop.
    # We limit epochs to 1 and batch_size to 2 for speed.
    run_training(
        train_meta_path=demo_train_path,
        val_meta_path=demo_val_path,
        test_meta_path=demo_test_path,
        submission_path=SUBMISSION_PATH,
        cache_dir=CACHE_DIR,
        epochs=1,
        batch_size=2,
        lr=1e-4,
        seed=42,
        load_cached_data=True,  # Use the data we just cached
    )

    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training pipeline verified.")

    # ==========================================
    # 7. Demonstrate Library: Predict
    # ==========================================
    print("\n[Step 6] Verifying library.predict...")

    # Generate submission using the model trained in the previous step
    generate_submission(
        test_meta_path=demo_test_path,
        model_path=best_model_path,
        submission_path=SUBMISSION_PATH,
        cache_dir=CACHE_DIR,
        batch_size=2,
        num_workers=0,
        load_cached_data=True,
    )

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(SUBMISSION_PATH)
    print("Submission File Head:")
    print(sub_df.head())

    assert (
        len(sub_df) == subset_size
    ), f"Expected {subset_size} predictions, got {len(sub_df)}"
    assert "BraTS21ID" in sub_df.columns
    assert "MGMT_value" in sub_df.columns

    print("Inference pipeline verified.")

    print("\n" + "=" * 40)
    print(" DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
