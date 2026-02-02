import os
import pandas as pd
import torch
import numpy as np

# Import components from the provided library files
from library.utils import set_seed, get_device
from library.data_loader import BraTSDataset, get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import run_training, generate_submission

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("Initializing configuration...")
    set_seed(42)
    device = get_device()
    print(f"Compute device selected: {device}")

    # Define paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    CACHE_PATH = os.path.join(WORKING_DIR, "roi_cache.parquet")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Optimization: Pre-generate ROI Cache
    # -------------------------------------------------------------------------
    # The data loader normally scans all DICOM files to find the 'best' slice
    # (ROI). To save time for this demonstration, we pre-populate the cache
    # with a default ratio (0.5, middle of the scan) for all subjects.
    print("Pre-generating ROI cache to bypass heavy I/O initialization...")

    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Collect all unique IDs across splits
    all_ids = pd.concat(
        [train_df["BraTS21ID"], val_df["BraTS21ID"], test_df["BraTS21ID"]]
    ).unique()

    # Create cache dataframe
    cache_data = [{"BraTS21ID": uid, "anchor_ratio": 0.5} for uid in all_ids]
    pd.DataFrame(cache_data).to_parquet(CACHE_PATH)
    print(f"Cache file created at {CACHE_PATH} with {len(cache_data)} entries.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loader Logic
    # -------------------------------------------------------------------------
    print("\n--- Verifying Data Loader Components ---")

    # Create a small subset for quick verification
    subset_df = train_df.head(4).copy()

    # Instantiate Dataset
    ds = BraTSDataset(
        df=subset_df, root_dir="./input", phase="train", load_cached_data=True
    )

    # Verify single item retrieval
    img_tensor, target = ds[0]
    print(f"Single item shape: {img_tensor.shape}")
    print(f"Target value: {target}")

    # Assertions
    # Shape should be (Channels, Height, Width).
    # Channels = 4 modalities * 3 slices = 12.
    assert img_tensor.shape == (
        12,
        224,
        224,
    ), f"Expected (12, 224, 224), got {img_tensor.shape}"
    assert isinstance(target, torch.Tensor), "Target should be a torch.Tensor"

    # Instantiate DataLoader
    loader = get_dataloader(
        subset_df,
        root_dir="./input",
        phase="train",
        batch_size=2,
        load_cached_data=True,
    )

    # Verify batch retrieval
    batch_imgs, batch_targets = next(iter(loader))
    print(f"Batch images shape: {batch_imgs.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")

    assert batch_imgs.shape == (2, 12, 224, 224), "Batch image shape mismatch"
    assert batch_targets.shape == (2,), "Batch target shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    # Instantiate model (using pretrained=False for speed in this check)
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.2, pretrained=False)
    model = model.to(device)
    model.eval()

    # Perform forward pass with the batch from previous step
    with torch.no_grad():
        inputs = batch_imgs.to(device)
        outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")

    # Output should be (Batch_Size, Num_Classes) -> (2, 1)
    assert outputs.shape == (2, 1), f"Expected output (2, 1), got {outputs.shape}"

    # -------------------------------------------------------------------------
    # 5. Execute Training Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Executing Training Pipeline (1 Epoch) ---")

    # run_training handles the full loop: loading metadata, training, validation,
    # and saving the best model to ./working/idea_14/best_model.pth
    best_auc = run_training(
        epochs=1,  # Limit to 1 epoch for demonstration speed
        batch_size=32,  # Standard batch size
        lr=1e-3,
        patience=1,
        seed=42,
        load_cached_data=True,
    )

    print(f"Training finished. Best Validation AUC: {best_auc:.4f}")

    # Verify model file was saved
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model checkpoint was not saved."

    # -------------------------------------------------------------------------
    # 6. Generate Submission
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    # Generates predictions for the test set using TTA
    generate_submission(batch_size=32, load_cached_data=True)

    # Verify submission file
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission file created successfully at {submission_path}")
        print(f"Rows: {len(sub_df)}")
        print(sub_df.head())

        # Validate row count matches test set
        assert len(sub_df) == len(
            test_df
        ), f"Submission rows ({len(sub_df)}) do not match test set ({len(test_df)})"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nScript completed successfully.")
