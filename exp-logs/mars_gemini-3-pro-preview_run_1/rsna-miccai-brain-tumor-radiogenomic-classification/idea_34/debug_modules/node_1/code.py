import os
import pandas as pd
import torch
import numpy as np
import sys

# Import library modules
import library.config as config
import library.utils as utils
import library.roi_processing as roi_processing
import library.data_loader as data_loader
import library.network as network
import library.trainer as trainer


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Reproducibility
    utils.seed_everything(42)

    # 2. Create Data Subsets for Speed
    # We create small metadata files (4 samples each) to allow the ROI processing
    # and training to finish in seconds/minutes rather than hours.
    subset_dir = "./working/subset_metadata"
    os.makedirs(subset_dir, exist_ok=True)

    subset_train_path = os.path.join(subset_dir, "train_metadata.csv")
    subset_val_path = os.path.join(subset_dir, "val_metadata.csv")
    subset_test_path = os.path.join(subset_dir, "test_metadata.csv")

    print("Creating metadata subsets (4 samples each)...")
    # Load original metadata
    df_orig_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_orig_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_orig_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Save head(4) to new paths
    df_orig_train.head(4).to_csv(subset_train_path, index=False)
    df_orig_val.head(4).to_csv(subset_val_path, index=False)
    df_orig_test.head(4).to_csv(subset_test_path, index=False)

    # 3. Patch Configuration across Modules
    # Since modules import variables directly, we must patch them in the module namespace
    print("Patching configuration for rapid execution...")

    # Patch Config Module
    config.TRAIN_METADATA_PATH = subset_train_path
    config.VAL_METADATA_PATH = subset_val_path
    config.TEST_METADATA_PATH = subset_test_path
    config.BATCH_SIZE = 2
    config.NUM_EPOCHS = 1

    # Patch ROI Processing Module
    roi_processing.TRAIN_METADATA_PATH = subset_train_path
    roi_processing.VAL_METADATA_PATH = subset_val_path
    roi_processing.TEST_METADATA_PATH = subset_test_path

    # Patch Data Loader Module
    data_loader.BATCH_SIZE = 2

    # Patch Trainer Module
    trainer.BATCH_SIZE = 2
    trainer.NUM_EPOCHS = 1
    trainer.EARLY_STOPPING_PATIENCE = 1

    # 4. Clear Cache
    # Remove existing ROI cache to ensure our new subset is processed
    cache_files = [
        "roi_cache_train.parquet",
        "roi_cache_val.parquet",
        "roi_cache_test.parquet",
    ]
    for f in cache_files:
        p = os.path.join(config.WORK_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    # 5. Demonstrate ROI Processing
    print("\n--- Step 1: ROI Processing ---")
    # This reads DICOMs, finds brain tissue bounds, and selects slices
    df_train, df_val, df_test = roi_processing.generate_roi_cache(
        load_cached_data=False
    )

    print(f"Processed Train Size: {len(df_train)}")
    print(f"Processed Val Size: {len(df_val)}")
    print(f"Processed Test Size: {len(df_test)}")

    assert len(df_train) == 4, "Train subset size mismatch"
    assert len(df_val) == 4, "Val subset size mismatch"
    assert len(df_test) == 4, "Test subset size mismatch"

    # 6. Demonstrate Data Loading
    print("\n--- Step 2: Data Loading ---")
    # Initialize loaders (will use the cache we just generated)
    loaders = data_loader.get_dataloaders(load_cached_data=True)
    train_loader = loaders["train"]

    # Fetch a single batch
    images, targets = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    # Verify Shapes: (Batch_Size, Channels, H, W) -> (2, 9, 224, 224)
    expected_shape = (2, 9, 224, 224)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
        )

    if targets.shape != (2,):
        raise AssertionError(
            f"Target shape mismatch. Expected (2,), got {targets.shape}"
        )

    # 7. Demonstrate Network Architecture
    print("\n--- Step 3: Network Architecture ---")
    model = network.VRAWIVModel()
    model.eval()

    # Forward pass on CPU (or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    images = images.to(device)

    with torch.no_grad():
        output = model(images)

    print(f"Model Output Shape: {output.shape}")

    if output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
        )

    # 8. Demonstrate Training Pipeline
    print("\n--- Step 4: Training & Inference ---")
    # Initialize Trainer
    demo_trainer = trainer.Trainer()

    # Run Training (Fit)
    print("Running training loop (1 Epoch)...")
    demo_trainer.fit(loaders["train"], loaders["val"])

    # Verify Model Checkpoint
    if not os.path.exists(demo_trainer.best_model_path):
        raise AssertionError("Best model checkpoint was not saved.")
    print("Training complete. Checkpoint verified.")

    # Run Inference (Predict)
    print("Running inference on test set...")
    demo_trainer.predict_and_submit(loaders["test"])

    # Verify Submission File
    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not generated.")

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission File Content (Head):")
    print(sub_df.head())

    if len(sub_df) != 4:
        raise AssertionError(
            f"Submission length mismatch. Expected 4, got {len(sub_df)}"
        )

    if list(sub_df.columns) != ["BraTS21ID", "MGMT_value"]:
        raise AssertionError(f"Submission columns mismatch. Got {list(sub_df.columns)}")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
