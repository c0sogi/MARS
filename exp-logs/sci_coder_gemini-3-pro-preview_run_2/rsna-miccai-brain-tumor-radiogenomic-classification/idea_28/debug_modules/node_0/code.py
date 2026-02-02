import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.data import MGMTDataset, get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import run_training


def run_demo():
    print("=== Starting Glioblastoma MGMT Prediction Demo ===")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Define demo-specific paths in the working directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use our demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_FILE_PATH = os.path.join(DEMO_DIR, "roi_cache_demo.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Override Training Hyperparameters for speed
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EARLY_STOPPING_PATIENCE = 2
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo

    # Ensure reproducibility
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Save Path: {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 2. Data Preparation (Subsetting)
    # --------------------------------------------------------------------------
    print("\n[2] Creating data subsets for demonstration...")

    # Load original metadata
    orig_train_df = pd.read_csv("./metadata/train.csv")
    orig_val_df = pd.read_csv("./metadata/val.csv")
    orig_test_df = pd.read_csv("./metadata/test.csv")

    # Create tiny subsets (e.g., 8 samples for train, 4 for val, 4 for test)
    # We select samples that definitely exist to avoid file errors
    subset_train = orig_train_df.head(8).copy()
    subset_val = orig_val_df.head(4).copy()
    subset_test = orig_test_df.head(4).copy()

    # Save subsets to the demo directory
    subset_meta_dir = os.path.join(DEMO_DIR, "subset_metadata")
    os.makedirs(subset_meta_dir, exist_ok=True)

    demo_train_path = os.path.join(subset_meta_dir, "train.csv")
    demo_val_path = os.path.join(subset_meta_dir, "val.csv")
    demo_test_path = os.path.join(subset_meta_dir, "test.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)
    subset_test.to_csv(demo_test_path, index=False)

    # Point Config to these new metadata files
    Config.TRAIN_CSV = demo_train_path
    Config.VAL_CSV = demo_val_path
    Config.TEST_CSV = demo_test_path

    print(f"Created subset train: {len(subset_train)} samples")
    print(f"Created subset val: {len(subset_val)} samples")

    # --------------------------------------------------------------------------
    # 3. Component Verification: Dataset & Loader
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoader...")

    # Initialize Dataset
    # This will trigger ROI cache generation for the subset
    ds = MGMTDataset(subset_train, transform=None, load_cached_roi=False)

    # Check length
    assert len(ds) == 8, f"Dataset length mismatch. Expected 8, got {len(ds)}"

    # Check item structure
    img_tensor, label_tensor = ds[0]

    # Expected shape: (C, H, W) -> (12, 224, 224)
    # 12 channels = 4 modalities * 3 slices
    expected_channels = Config.NUM_MODALITIES * Config.NUM_SLICES_PER_MODALITY
    print(f"Sample Tensor Shape: {img_tensor.shape}")

    assert img_tensor.dim() == 3, "Image tensor must be 3D (C, H, W)"
    assert (
        img_tensor.shape[0] == expected_channels
    ), f"Expected {expected_channels} channels, got {img_tensor.shape[0]}"
    assert img_tensor.shape[1] == Config.IMG_SIZE, f"Expected height {Config.IMG_SIZE}"
    assert img_tensor.shape[2] == Config.IMG_SIZE, f"Expected width {Config.IMG_SIZE}"
    assert isinstance(label_tensor, torch.Tensor), "Label must be a torch Tensor"

    # Check DataLoader
    loader = get_dataloader(
        subset_train, phase="train", batch_size=Config.BATCH_SIZE, num_workers=0
    )
    batch_imgs, batch_labels = next(iter(loader))

    print(f"Batch Tensor Shape: {batch_imgs.shape}")
    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_imgs.shape[1] == expected_channels, "Channel count mismatch in batch"

    # --------------------------------------------------------------------------
    # 4. Component Verification: Model
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = AsymmetricEfficientNet().to(device)
    model.eval()

    with torch.no_grad():
        # Pass the batch we just loaded
        output = model(batch_imgs.to(device))

    print(f"Model Output Shape: {output.shape}")

    # Output should be (Batch_Size, 1) - raw logits
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {output.shape}"

    print("Model verification successful.")

    # --------------------------------------------------------------------------
    # 5. Training Simulation
    # --------------------------------------------------------------------------
    print("\n[5] Running Training Simulation...")

    # run_training() reads from Config.TRAIN_CSV/VAL_CSV which we updated
    run_training()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created!"
    print("Training simulation complete. Checkpoint saved.")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_CSV)

    # Create Test Loader
    test_loader = get_dataloader(
        df_test, phase="test", batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Load Best Model
    device = torch.device(Config.DEVICE)
    loaded_model = AsymmetricEfficientNet().to(device)
    loaded_model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    )
    loaded_model.eval()

    predictions = []
    ids = []

    print("Predicting on test subset...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # Note: labels in test loader are dummy 0.0s
            inputs = inputs.to(device)

            # Forward pass
            logits = loaded_model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)

            # Get IDs for this batch
            # We need to map back to BraTS21ID.
            # Since loader is sequential and not shuffled (phase='test'), we can slice the dataframe.
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + inputs.size(0)
            batch_ids = df_test.iloc[start_idx:end_idx]["BraTS21ID"].values
            ids.extend(batch_ids)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Verify predictions are probabilities
    assert submission_df["MGMT_value"].min() >= 0.0
    assert submission_df["MGMT_value"].max() <= 1.0

    print("Sample Predictions:")
    print(submission_df.head())

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
