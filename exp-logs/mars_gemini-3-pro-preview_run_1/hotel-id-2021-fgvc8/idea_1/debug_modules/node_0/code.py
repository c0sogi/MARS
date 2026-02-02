import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library
from library.config import Config, seed_everything, setup_directories
from library.utils import HotelIdLabelEncoder, calculate_map5, get_label_encoder
from library.dataset import HotelDataset, get_transforms
from library.model import HotelResNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_utils():
    print("\n--- 1. Demonstrating Utility Functions ---")

    # A. Test HotelIdLabelEncoder
    print("Testing HotelIdLabelEncoder...")
    dummy_ids = [1001, 2002, 1001, 3003, 2002]
    encoder_path = os.path.join(Config.WORKING_DIR, "test_encoder.npy")

    encoder = HotelIdLabelEncoder()
    encoder.fit(dummy_ids)

    # Check classes
    expected_classes = np.array([1001, 2002, 3003])
    assert np.array_equal(
        encoder.classes_, expected_classes
    ), "Encoder classes mismatch"

    # Check transform
    transformed = encoder.transform(dummy_ids)
    expected_indices = np.array([0, 1, 0, 2, 1])
    assert np.array_equal(transformed, expected_indices), "Encoder transform mismatch"

    # Check inverse transform
    inversed = encoder.inverse_transform(transformed)
    assert np.array_equal(inversed, dummy_ids), "Encoder inverse transform mismatch"

    # Check save/load
    encoder.save(encoder_path)
    assert os.path.exists(encoder_path), "Encoder file not saved"

    encoder_loaded = HotelIdLabelEncoder()
    encoder_loaded.load(encoder_path)
    assert np.array_equal(
        encoder_loaded.classes_, encoder.classes_
    ), "Loaded encoder classes mismatch"

    # Cleanup dummy encoder
    if os.path.exists(encoder_path):
        os.remove(encoder_path)

    print("HotelIdLabelEncoder verified successfully.")

    # B. Test MAP@5 Calculation
    print("Testing calculate_map5...")
    # Scenario:
    # Item 0: Target 10. Preds: [10, 20, 30, 40, 50]. Rank 0 (1st). AP = 1/1 = 1.0
    # Item 1: Target 20. Preds: [10, 30, 20, 40, 50]. Rank 2 (3rd). AP = 1/3 = 0.333...
    # Item 2: Target 99. Preds: [10, 20, 30, 40, 50]. Not in top 5. AP = 0.0

    targets = [10, 20, 99]
    preds = [[10, 20, 30, 40, 50], [10, 30, 20, 40, 50], [10, 20, 30, 40, 50]]

    score = calculate_map5(preds, targets)
    expected_score = (1.0 + (1.0 / 3.0) + 0.0) / 3.0

    assert np.isclose(
        score, expected_score
    ), f"MAP@5 calculation incorrect. Got {score}, expected {expected_score}"
    print(f"MAP@5 verified successfully. Score: {score:.4f}")


def demo_dataset():
    print("\n--- 2. Demonstrating Dataset Pipeline ---")

    # Load a small subset of metadata for testing
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH).head(10)

    # Test Transforms
    print("Testing Transforms...")
    transform = get_transforms(phase="train", image_size=Config.IMAGE_SIZE)
    assert transform is not None, "Failed to get transforms"

    # Test Dataset instantiation
    print("Instantiating HotelDataset (Train phase)...")
    # We use a temporary encoder path to avoid messing with the main training flow
    temp_encoder_path = os.path.join(Config.WORKING_DIR, "dataset_test_encoder.npy")

    # Ensure we fit the encoder on this small subset for the demo
    encoder = HotelIdLabelEncoder()
    encoder.fit(train_df["hotel_id"].values)

    dataset = HotelDataset(
        df=train_df,
        phase="train",
        transform=transform,
        label_encoder=encoder,
        data_root=Config.INPUT_DIR,
    )

    assert len(dataset) == 10, "Dataset length mismatch"

    # Test __getitem__
    print("Fetching a sample item...")
    sample = dataset[0]

    # Check keys
    assert "image" in sample, "Missing 'image' key"
    assert "target" in sample, "Missing 'target' key"
    assert "image_id" in sample, "Missing 'image_id' key"

    # Check shapes
    img_tensor = sample["image"]
    target = sample["target"]

    assert isinstance(img_tensor, torch.Tensor), "Image is not a tensor"
    assert img_tensor.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Unexpected image shape: {img_tensor.shape}"
    assert isinstance(target, torch.Tensor), "Target is not a tensor"

    print("Dataset pipeline verified successfully.")


def demo_model():
    print("\n--- 3. Demonstrating Model Architecture ---")

    # Initialize model
    # We use a small number of classes for the demo to save memory/init time,
    # though ResNet18 is light enough.
    n_classes_demo = 50
    model = HotelResNet(n_classes=n_classes_demo, pretrained=False)

    # Move to configured device
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )

    print(f"Running forward pass with input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size, N_Classes)
    expected_shape = (batch_size, n_classes_demo)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"

    print("Model forward pass verified successfully.")


def demo_training_and_inference():
    print("\n--- 4. Demonstrating Trainer (Fit & Predict) ---")

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    # We use the actual metadata paths but rely on Config.DEBUG=True to limit the data size
    print("Starting training loop (Debug Mode)...")
    trainer.fit(
        train_metadata_path=Config.TRAIN_METADATA_PATH,
        val_metadata_path=Config.VAL_METADATA_PATH,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        num_workers=Config.NUM_WORKERS,
    )

    # Check if best model was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint not found after training"
    print("Training loop completed and model saved.")

    # Run Prediction
    print("Starting inference on test set...")
    trainer.predict(
        test_metadata_path=Config.TEST_METADATA_PATH,
        submission_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check format
    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission columns incorrect"

    # Check prediction format (space delimited string)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction is not a string"
    assert (
        len(sample_pred.split()) == 5
    ), f"Prediction does not contain 5 IDs: {sample_pred}"

    print("Inference and submission generation verified successfully.")


if __name__ == "__main__":
    # 0. Setup
    seed_everything(Config.SEED)
    setup_directories()

    # Override Config for Speed/Demo purposes
    print("Configuring for fast demonstration...")
    Config.EPOCHS = 1  # Train only 1 epoch
    Config.DEBUG = True  # Use debug mode (subsets data)
    Config.DEBUG_SAMPLE_SIZE = 100  # Very small subset for speed
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.PRETRAINED = False  # Skip downloading weights for speed if not cached

    # Print modified config
    # Config.print_config()

    try:
        # 1. Utils
        demo_utils()

        # 2. Dataset
        demo_dataset()

        # 3. Model
        demo_model()

        # 4. Trainer (Full Loop)
        demo_training_and_inference()

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")
        # Print traceback for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
