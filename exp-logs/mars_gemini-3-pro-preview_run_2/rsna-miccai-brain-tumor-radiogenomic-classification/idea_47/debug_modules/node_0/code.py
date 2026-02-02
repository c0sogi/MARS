import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, read_dicom_image, get_logger
from library.data import BrainTumorDataset
from library.model import AsymmetricEfficientNet
from library.train import train_one_epoch, validate
from library.predict import predict_submission


def run_demo():
    print("--- Starting Library Demonstration ---")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # --------------------------------------------------------------------------
    # We override Config parameters to run a fast demo on a small subset
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")

    # Monkey-patch Config to use demo paths and settings
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CSV = os.path.join(demo_dir, "demo_train.csv")
    Config.VAL_CSV = os.path.join(demo_dir, "demo_val.csv")
    Config.TEST_CSV = os.path.join(demo_dir, "demo_test.csv")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Reduce compute load for demo
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo
    Config.PATIENCE = 1

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Prepare Data Subsets
    # --------------------------------------------------------------------------
    print("\n[Step 1] Preparing Data Subsets...")

    # Load original metadata
    orig_train_df = pd.read_csv("./metadata/train.csv")
    orig_val_df = pd.read_csv("./metadata/val.csv")
    orig_test_df = pd.read_csv("./metadata/test.csv")

    # Sample small subsets (e.g., 4 samples for train, 2 for val, 2 for test)
    # Ensure we pick samples that actually exist on disk (metadata should be correct)
    demo_train_df = orig_train_df.head(4).copy()
    demo_val_df = orig_val_df.head(2).copy()
    demo_test_df = orig_test_df.head(2).copy()

    # Save to demo directory
    demo_train_df.to_csv(Config.TRAIN_CSV, index=False)
    demo_val_df.to_csv(Config.VAL_CSV, index=False)
    demo_test_df.to_csv(Config.TEST_CSV, index=False)

    assert os.path.exists(Config.TRAIN_CSV), "Demo train CSV not created"
    print(
        f"Created subset CSVs: Train={len(demo_train_df)}, Val={len(demo_val_df)}, Test={len(demo_test_df)}"
    )

    # --------------------------------------------------------------------------
    # 3. Verify Image Reading
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Image Reader...")

    # Pick a sample path from the dataframe
    sample_row = demo_train_df.iloc[0]
    # Construct a full path to a specific DICOM file
    # We need to find a file that exists. The metadata points to directories.
    flair_dir = os.path.join(Config.INPUT_DIR, sample_row["path_FLAIR"])
    files = sorted(os.listdir(flair_dir))
    if files:
        sample_dcm_path = os.path.join(flair_dir, files[len(files) // 2])

        # Test read_dicom_image
        img = read_dicom_image(
            sample_dcm_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE)
        )

        assert isinstance(img, np.ndarray), "Image is not a numpy array"
        assert img.shape == (
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Incorrect shape: {img.shape}"
        assert img.dtype == np.float32, f"Incorrect dtype: {img.dtype}"
        print(f"Successfully read image: {sample_dcm_path} -> Shape {img.shape}")
    else:
        print("Warning: No files found in sample directory to test image reader.")

    # --------------------------------------------------------------------------
    # 4. Verify Dataset & ROI Caching
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying BrainTumorDataset...")

    # Initialize dataset (this triggers ROI calculation and caching)
    train_dataset = BrainTumorDataset(
        demo_train_df, phase="train", load_cached_data=True
    )

    # Verify cache file creation
    cache_path = os.path.join(Config.WORKING_DIR, "roi_cache.parquet")
    assert os.path.exists(cache_path), "ROI cache file was not created"

    # Verify __len__
    assert len(train_dataset) == len(demo_train_df)

    # Verify __getitem__
    volume, label = train_dataset[0]

    # Expected shape: (Channels, Height, Width)
    # Channels = 4 modalities * 3 slices = 12
    expected_shape = (Config.INPUT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)

    assert torch.is_tensor(volume), "Output volume is not a tensor"
    assert (
        volume.shape == expected_shape
    ), f"Volume shape mismatch. Got {volume.shape}, expected {expected_shape}"
    assert torch.is_tensor(label), "Label is not a tensor"

    print(f"Dataset verification passed. Volume shape: {volume.shape}")

    # --------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying AsymmetricEfficientNet Model...")

    model = AsymmetricEfficientNet()
    model.to(device)

    # Check first layer modifications
    first_conv = model.backbone.features[0][0]
    assert (
        first_conv.in_channels == Config.INPUT_CHANNELS
    ), f"Model input channels wrong: {first_conv.in_channels}"
    assert first_conv.groups == len(
        Config.MODALITIES
    ), f"Model groups wrong: {first_conv.groups}"

    # Test forward pass with the sample volume
    # Add batch dimension: (1, 12, 224, 224)
    input_tensor = volume.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (1, 1), f"Model output shape wrong: {output.shape}"
    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 6. Verify Training Loop
    # --------------------------------------------------------------------------
    print("\n[Step 5] Verifying Training Step...")

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch
    initial_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    assert isinstance(initial_loss, float), "Train loss is not a float"
    assert not np.isnan(initial_loss), "Train loss is NaN"
    print(f"Training step complete. Loss: {initial_loss:.4f}")

    # Verify Validation
    val_dataset = BrainTumorDataset(demo_val_df, phase="val", load_cached_data=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation step complete. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Save dummy model for prediction step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint not saved"

    # --------------------------------------------------------------------------
    # 7. Verify Prediction Pipeline
    # --------------------------------------------------------------------------
    print("\n[Step 6] Verifying Prediction Pipeline...")

    # predict_submission reads Config.TEST_CSV and Config.BEST_MODEL_PATH
    # and writes to Config.SUBMISSION_PATH
    predict_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_df) == len(demo_test_df), "Submission row count mismatch"
    assert "BraTS21ID" in submission_df.columns, "BraTS21ID column missing"
    assert "MGMT_value" in submission_df.columns, "MGMT_value column missing"

    # Check probabilities are valid
    probs = submission_df["MGMT_value"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("Prediction pipeline successful.")
    print(submission_df.head())

    print("\n--- All Demonstrations Passed Successfully ---")


if __name__ == "__main__":
    run_demo()
