import os
import shutil
import pandas as pd
import numpy as np
import torch
import library.utils as utils
import library.data as data_lib
import library.model as model_lib
import library.train as train_lib
import library.inference as inference_lib

# ==========================================
# Configuration
# ==========================================
DEMO_DIR = "./working/demo_run"
DEMO_METADATA_DIR = "./working/demo_metadata"
DEMO_CACHE_DIR = "./working/demo_cache"
ORIGINAL_METADATA_DIR = "./metadata"

# Ensure clean state
for d in [DEMO_DIR, DEMO_METADATA_DIR, DEMO_CACHE_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


# ==========================================
# 1. Data Subsetting for Fast Execution
# ==========================================
def create_subset_metadata():
    """
    Creates a small subset of the metadata files to allow the demo
    to run quickly without processing the entire dataset.
    """
    print(f"Creating metadata subset in {DEMO_METADATA_DIR}...")

    splits = ["train", "val", "test"]
    subset_size = 10  # Only process 10 patients per split

    for split in splits:
        src_path = os.path.join(ORIGINAL_METADATA_DIR, f"{split}.parquet")
        dst_path = os.path.join(DEMO_METADATA_DIR, f"{split}.parquet")

        if os.path.exists(src_path):
            df = pd.read_parquet(src_path)
            # Take a small subset
            df_subset = df.head(subset_size).copy()
            df_subset.to_parquet(dst_path, index=False)
            print(f"  - {split}: Saved {len(df_subset)} samples.")
        else:
            print(f"  - {split}: Source file not found, skipping.")


# ==========================================
# 2. Monkey-Patching Library Config
# ==========================================
def configure_library_paths():
    """
    Redirects the library's internal path constants to our demo directories.
    This ensures we don't touch the main experiment's cache or data.
    """
    print("Configuring library to use demo paths...")
    data_lib.METADATA_DIR = DEMO_METADATA_DIR
    data_lib.CACHE_DIR = DEMO_CACHE_DIR
    # We also update the cache dir in train/inference if they reference it,
    # but based on the code provided, they call data_lib functions which use the patched var.


# ==========================================
# 3. Model Demonstration
# ==========================================
def demonstrate_model_architecture():
    print("\n=== Demonstrating Model Architecture ===")

    device = utils.get_device()
    print(f"Device: {device}")

    # Instantiate model
    # We use a small backbone (efficientnet_b0) as defined in the library
    model = model_lib.SDVNet(model_name="efficientnet_b0", pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy inputs
    # Shape: (Batch, Channels, Height, Width)
    # Channels = 64 (16 slices * 4 modalities)
    batch_size = 2
    dummy_even = torch.randn(batch_size, 64, 256, 256).to(device)
    dummy_odd = torch.randn(batch_size, 64, 256, 256).to(device)

    print(f"Input shape (Even Stream): {dummy_even.shape}")
    print(f"Input shape (Odd Stream):  {dummy_odd.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_even, dummy_odd)

    print(f"Output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
    print("Model forward pass successful.")


# ==========================================
# 4. Training Demonstration
# ==========================================
def demonstrate_training_loop():
    print("\n=== Demonstrating Training Loop ===")

    # Run training for a minimal number of epochs
    # load_cached_data=False forces the data processing logic to run on our subset
    best_auc = train_lib.run_training(
        epochs=2,
        batch_size=4,
        learning_rate=1e-4,
        patience=2,
        save_dir=DEMO_DIR,
        load_cached_data=False,
    )

    # Verify artifacts
    model_path = os.path.join(DEMO_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Training failed to save 'best_model.pth'"

    print(f"Training demo complete. Best AUC: {best_auc}")
    print(f"Model saved to: {model_path}")


# ==========================================
# 5. Inference Demonstration
# ==========================================
def demonstrate_inference():
    print("\n=== Demonstrating Inference ===")

    model_path = os.path.join(DEMO_DIR, "best_model.pth")
    submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Run inference
    inference_lib.predict_submission(
        model_path=model_path,
        save_path=submission_path,
        batch_size=4,
        load_cached_data=True,  # Can use cache now as it was generated during training setup
        device=utils.get_device(),
    )

    # Verify output
    assert os.path.exists(
        submission_path
    ), "Inference failed to create submission file."

    df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    assert "BraTS21ID" in df.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df.columns, "Missing MGMT_value column"
    assert len(df) > 0, "Submission file is empty"

    print("Inference demo successful.")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    utils.seed_everything(42)
    create_subset_metadata()
    configure_library_paths()

    # 2. Run Demos
    demonstrate_model_architecture()
    demonstrate_training_loop()
    demonstrate_inference()

    print("\nAll demonstrations completed successfully.")
