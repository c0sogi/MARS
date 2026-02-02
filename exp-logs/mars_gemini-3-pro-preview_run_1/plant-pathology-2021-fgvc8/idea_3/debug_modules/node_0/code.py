import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleDataset, get_transforms
from library.model import AppleDiseaseSwinModel
from library.train import run_training, run_inference


def verify_dataset_logic():
    print("\n--- Verifying Dataset Logic ---")

    # Load metadata
    df_train = pd.read_csv(Config.train_metadata_path)

    # Create a small subset for testing
    df_subset = df_train.head(16).reset_index(drop=True)

    # Instantiate Dataset
    dataset = AppleDataset(
        df=df_subset,
        mode="train",
        transform=get_transforms(mode="train", img_size=Config.img_size),
    )

    print(f"Dataset length: {len(dataset)}")

    # Fetch one sample
    image, label = dataset[0]

    # Assertions
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label example: {label}")

    # Check Image Dimensions: (Channels, Height, Width)
    assert image.shape == (
        3,
        Config.img_size,
        Config.img_size,
    ), f"Expected image shape (3, {Config.img_size}, {Config.img_size}), got {image.shape}"

    # Check Label Dimensions: (Num_Classes,)
    assert label.shape == (
        Config.num_classes,
    ), f"Expected label shape ({Config.num_classes},), got {label.shape}"

    # Check Label Dtype
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    assert label.dtype == torch.float32, "Label dtype should be float32"

    # Check Multi-hot encoding (values should be 0.0 or 1.0)
    unique_vals = torch.unique(label)
    for val in unique_vals:
        assert val.item() in [0.0, 1.0], f"Label contains invalid value: {val.item()}"

    print("Dataset logic verified successfully.")


def verify_model_logic():
    print("\n--- Verifying Model Logic ---")

    device = torch.device(
        "cpu"
    )  # Use CPU for quick logic check to avoid GPU overhead if busy

    # Instantiate Model
    model = AppleDiseaseSwinModel(pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy input batch: (Batch_Size, Channels, Height, Width)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, Config.img_size, Config.img_size).to(
        device
    )

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        batch_size,
        Config.num_classes,
    ), f"Expected output shape ({batch_size}, {Config.num_classes}), got {logits.shape}"

    print("Model logic verified successfully.")


def verify_training_pipeline():
    print("\n--- Verifying Training Pipeline ---")

    # Run training with debug mode enabled and minimal epochs for speed
    # debug=True uses Config.debug_sample_size (100 samples)
    run_training(
        debug=True,
        epochs=1,
        batch_size=8,
        num_workers=2,  # Reduce workers for small debug run
    )

    # Verify outputs exist
    expected_model_path = Config.model_save_path
    expected_log_path = os.path.join(Config.working_dir, "train_log.txt")

    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to produce model at {expected_model_path}"
        )

    if not os.path.exists(expected_log_path):
        raise FileNotFoundError(
            f"Training failed to produce log at {expected_log_path}"
        )

    print(f"Model saved at: {expected_model_path}")
    print("Training pipeline verified successfully.")


def verify_inference_pipeline():
    print("\n--- Verifying Inference Pipeline ---")

    # Run inference
    # This relies on the model saved in the previous step
    run_inference(batch_size=8, num_workers=2)

    # Verify submission file
    expected_sub_path = Config.submission_path

    if not os.path.exists(expected_sub_path):
        raise FileNotFoundError(
            f"Inference failed to produce submission at {expected_sub_path}"
        )

    # Load submission and check format
    df_sub = pd.read_csv(expected_sub_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Assertions
    assert "image" in df_sub.columns, "Submission missing 'image' column"
    assert "labels" in df_sub.columns, "Submission missing 'labels' column"

    # Check against test metadata count
    df_test_meta = pd.read_csv(Config.test_metadata_path)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count ({len(df_sub)}) does not match test metadata ({len(df_test_meta)})"

    print("Inference pipeline verified successfully.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    seed_everything(Config.seed)

    try:
        # 1. Verify Dataset
        verify_dataset_logic()

        # 2. Verify Model
        verify_model_logic()

        # 3. Verify Training (Integration Test)
        verify_training_pipeline()

        # 4. Verify Inference (Integration Test)
        verify_inference_pipeline()

        print("\nAll demonstrations and verifications completed successfully.")

    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        # Re-raise to ensure the process exits with error code if needed
        raise e
