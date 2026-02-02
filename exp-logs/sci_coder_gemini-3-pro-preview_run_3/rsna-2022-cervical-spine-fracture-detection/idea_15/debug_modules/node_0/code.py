import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.preprocessor import Preprocessor
from library.dataset import CervicalSpineDataset, get_transforms
from library.model import ConvNeXtMIL
from library.loss import ImplicitWeightedLoss
from library.trainer import Trainer
from library.inference import predict_test_set


def main():
    # --- 1. Setup & Configuration Overrides ---
    # We override configuration parameters to ensure the demo runs quickly
    # and uses minimal resources.
    print("--- Configuring for Demo ---")
    seed_everything(42)

    Config.DEBUG_DATA_SIZE = 10  # Limit to 10 samples for train/val/test
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.PRETRAINED = False  # Disable weight download for speed/offline safety

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG_DATA_SIZE} samples")
    print(f"Device: {Config.DEVICE}")

    # --- 2. Preprocessing Demonstration ---
    print("\n--- Running Preprocessor ---")
    # The Preprocessor will read the metadata, load DICOMs for the first 10 studies,
    # window/resize them, and save .npy files to the cache directory.
    preprocessor = Preprocessor()
    preprocessor.run()

    # Verification: Check if cache files were created for the training set
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if Config.DEBUG_DATA_SIZE:
        train_df = train_df.iloc[: Config.DEBUG_DATA_SIZE]

    # Check the first available study in the subset
    sample_uid = train_df.iloc[0]["StudyInstanceUID"]
    expected_cache_path = os.path.join(Config.CACHE_DIR, f"{sample_uid}.npy")

    if os.path.exists(expected_cache_path):
        print(f"Verification Passed: Cache file created at {expected_cache_path}")
        # Verify shape of cached file
        vol = np.load(expected_cache_path)
        print(f"Cached Volume Shape: {vol.shape}")
        # Expected: (Depth, 224, 224)
        assert vol.ndim == 3 and vol.shape[1:] == (
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), f"Cached volume has incorrect spatial dimensions: {vol.shape}"
    else:
        print(
            f"Warning: Cache file {expected_cache_path} not found. Input DICOMs might be missing in this environment."
        )

    # --- 3. Dataset & DataLoader Demonstration ---
    print("\n--- Testing Dataset and DataLoader ---")
    # Instantiate dataset with the subset dataframe
    dataset = CervicalSpineDataset(
        metadata_df=train_df,
        images_dir=Config.TRAIN_IMAGES_DIR,
        transform=get_transforms(split="train"),
        split="train",
    )

    # Fetch one sample
    sample = dataset[0]
    image = sample["image"]
    targets = sample["targets"]
    uid = sample["study_uid"]

    print(f"Sample UID: {uid}")
    print(f"Image Tensor Shape: {image.shape}")
    print(f"Targets: {targets}")

    # Assertions
    # Shape should be (Num_Slices, Channels=3, H, W) -> (64, 3, 224, 224)
    expected_shape = (
        Config.NUM_SLICES,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert (
        image.shape == expected_shape
    ), f"Incorrect image tensor shape. Expected {expected_shape}, got {image.shape}"

    # Targets should be (Num_Classes,) -> (8,)
    assert targets.shape == (
        Config.NUM_CLASSES,
    ), f"Incorrect target shape. Expected ({Config.NUM_CLASSES},), got {targets.shape}"

    print("Verification Passed: Dataset output shapes are correct.")

    # --- 4. Model & Loss Demonstration ---
    print("\n--- Testing Model and Loss ---")
    device = get_device()

    # Initialize model (pretrained=False for demo speed)
    model = ConvNeXtMIL(
        model_name=Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        in_channels=Config.IN_CHANNELS,
    )
    model.to(device)
    model.eval()

    # Create a dummy batch of size 2
    batch_images = torch.stack([image, image]).to(device)  # (2, 64, 3, 224, 224)
    batch_targets = torch.stack([targets, targets]).to(device)  # (2, 8)

    print(f"Input Batch Shape: {batch_images.shape}")

    with torch.no_grad():
        logits = model(batch_images)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, 8), got {logits.shape}"

    # Test Loss Function
    criterion = ImplicitWeightedLoss()
    loss = criterion(logits, batch_targets)

    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Verification Passed: Model forward pass and loss calculation successful.")

    # --- 5. Training Loop Demonstration ---
    print("\n--- Running Trainer (1 Epoch) ---")
    # Initialize Trainer
    # The Trainer class will internally use the Config settings we modified (DEBUG_DATA_SIZE=10, EPOCHS=1)
    trainer = Trainer()

    # Run training
    trainer.fit()

    # Verify model checkpoint
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not saved after training."
    print("Verification Passed: Training loop completed and model saved.")

    # --- 6. Inference Demonstration ---
    print("\n--- Running Inference ---")
    # Run inference on test set (limited by Config.DEBUG_DATA_SIZE)
    predict_test_set(debug_size=Config.DEBUG_DATA_SIZE)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission DataFrame Shape: {sub_df.shape}")
    print("First 5 rows of submission:")
    print(sub_df.head())

    # Check for required columns
    assert (
        "row_id" in sub_df.columns and "fractured" in sub_df.columns
    ), "Submission missing required columns 'row_id' or 'fractured'."

    # Check that predictions are within valid probability range [0, 1]
    # Note: fillna(0.5) is used in inference, so no NaNs should exist.
    assert (
        sub_df["fractured"].min() >= 0.0 and sub_df["fractured"].max() <= 1.0
    ), "Submission contains probabilities outside [0, 1]."

    print("Verification Passed: Inference completed and valid submission generated.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
