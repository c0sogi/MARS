import os
import pandas as pd
import numpy as np
import torch
import cv2
from library.config import Config
from library.utils import (
    seed_everything,
    read_dicom_robust,
    normalize_channel,
    get_downsampled_max_anchor,
)
from library.data import (
    get_roi_cache,
    MGMTDataset,
    get_transforms,
    get_test_dataset,
    get_train_val_datasets,
)
from library.model import AsymmetricEfficientNet
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    seed_everything(42)

    # Ensure working directory exists (as defined in Config)
    # We use the one in Config which is ./working/idea_23
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Create Mini-Datasets for Speed
    # We load the full metadata but only use a tiny fraction for the demo
    # to ensure the script runs in < 1 minute.
    print("\n[Step 1] Creating Mini-Datasets...")

    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Take top 4 samples for each
    mini_train = train_full.head(4).copy()
    mini_val = val_full.head(4).copy()
    mini_test = test_full.head(4).copy()

    # Save to working directory
    path_mini_train = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    path_mini_val = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    path_mini_test = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    mini_train.to_csv(path_mini_train, index=False)
    mini_val.to_csv(path_mini_val, index=False)
    mini_test.to_csv(path_mini_test, index=False)

    print(f"Saved mini datasets to {Config.WORKING_DIR}")

    # 3. Monkey-Patch Config
    # This redirects the library functions to use our mini datasets
    Config.TRAIN_METADATA_PATH = path_mini_train
    Config.VAL_METADATA_PATH = path_mini_val
    Config.TEST_METADATA_PATH = path_mini_test
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data to avoid overhead

    # 4. Verify Utility Functions
    print("\n[Step 2] Verifying Utility Functions...")

    # Pick a sample file from the mini train set
    sample_subject = mini_train.iloc[0]
    flair_dir = os.path.join(Config.INPUT_DIR, sample_subject["path_FLAIR"])
    flair_files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]

    if len(flair_files) > 0:
        sample_dcm = os.path.join(flair_dir, flair_files[0])

        # Test read_dicom_robust
        img = read_dicom_robust(sample_dcm)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.float32
        print(f"  - read_dicom_robust: Success (Shape: {img.shape})")

        # Test normalize_channel
        norm_img = normalize_channel(img)
        assert norm_img.min() >= 0.0 and norm_img.max() <= 1.0
        print("  - normalize_channel: Success")

        # Test get_downsampled_max_anchor
        # This scans the directory, so it verifies the logic
        anchor_idx = get_downsampled_max_anchor(flair_dir)
        assert isinstance(anchor_idx, int)
        assert anchor_idx >= 0
        print(f"  - get_downsampled_max_anchor: Success (Anchor: {anchor_idx})")
    else:
        print("  - Warning: No DICOM files found in sample directory to test utils.")

    # 5. Verify Model Architecture
    print("\n[Step 3] Verifying Model Architecture...")
    model = AsymmetricEfficientNet()
    # Move to configured device
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Create dummy input (Batch=2, Channels=12, H=224, W=224)
    dummy_input = torch.randn(2, 12, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1)
    print("  - AsymmetricEfficientNet Forward Pass: Success")

    # 6. Run Training (Integration Test)
    print("\n[Step 4] Running Training Loop (Fast Mode)...")
    # We use run_training from library.train
    # We override epochs and batch_size for speed
    # load_cached_data=False ensures we compute ROI for our new mini dataset

    best_auc = run_training(load_cached_data=False, epochs=1, batch_size=2, patience=1)

    print(f"  - Training Complete. Best AUC: {best_auc}")
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved!"

    # 7. Inference and Submission
    print("\n[Step 5] Running Inference on Test Set...")

    # Load test dataset (uses the monkey-patched Config.TEST_METADATA_PATH)
    test_dataset = get_test_dataset(load_cached_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0
    )

    # Load best model
    model = AsymmetricEfficientNet().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Sigmoid to get probability
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)

    # Generate Submission DataFrame
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_dataset.df["BraTS21ID"], "MGMT_value": predictions}
    )

    print("  - Inference Complete. Sample predictions:")
    print(submission_df)

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"  - Submission saved to {submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
