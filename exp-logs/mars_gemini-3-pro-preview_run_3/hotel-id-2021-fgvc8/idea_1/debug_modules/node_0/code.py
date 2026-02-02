import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import from library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    seed_everything,
    LabelEncoder,
    calculate_map5,
    get_label_encoder,
)
from library.dataset import HotelDataset, get_transforms
from library.model import HotelResNet
from library.train import run_training
from library.inference import generate_submission


def main():
    print("=== Starting Hotel ID Pipeline Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Verify Utility Functions
    # ---------------------------------------------------------
    print("--- 1. Verifying Utilities ---")
    seed_everything(42)

    # Test LabelEncoder
    print("Testing LabelEncoder...")
    le = LabelEncoder()
    dummy_ids = [1001, 2002, 3003, 1001]
    le.fit(dummy_ids)

    # Check transform
    encoded = le.transform(dummy_ids)
    expected_encoded = np.array([0, 1, 2, 0])
    assert np.array_equal(
        encoded, expected_encoded
    ), f"LabelEncoder transform mismatch. Got {encoded}, expected {expected_encoded}"

    # Check inverse transform
    decoded = le.inverse_transform(encoded)
    assert np.array_equal(
        decoded, dummy_ids
    ), "LabelEncoder inverse_transform failed to restore original IDs."
    print("LabelEncoder passed.")

    # Test MAP@5 Calculation
    print("Testing MAP@5 Metric...")
    # Scenario:
    # Sample 0: Target 10. Preds [10, 11, 12, 13, 14] -> Rank 1 -> Score 1.0
    # Sample 1: Target 20. Preds [21, 20, 22, 23, 24] -> Rank 2 -> Score 0.5
    # Mean Score: 0.75
    targets = [10, 20]
    preds = [[10, 11, 12, 13, 14], [21, 20, 22, 23, 24]]
    score = calculate_map5(preds, targets)
    expected_score = 0.75
    assert np.isclose(
        score, expected_score
    ), f"MAP@5 calculation incorrect. Got {score}, expected {expected_score}"
    print("MAP@5 Metric passed.")

    # ---------------------------------------------------------
    # 2. Verify Dataset Loading
    # ---------------------------------------------------------
    print("\n--- 2. Verifying Dataset ---")
    # Load encoder based on actual metadata
    # We force reload to ensure logic runs, though cache is available
    encoder = get_label_encoder(Config.TRAIN_CSV, load_cached_data=False)

    # Initialize Dataset
    train_ds = HotelDataset(
        csv_path=Config.TRAIN_CSV,
        root_dir=Config.INPUT_DIR,
        label_encoder=encoder,
        transform=get_transforms("train"),
        is_test=False,
        load_cached_data=False,
    )

    print(f"Total training samples: {len(train_ds)}")

    # Check a single sample
    if len(train_ds) > 0:
        img, label = train_ds[0]
        print(f"Sample image shape: {img.shape}")
        print(f"Sample label index: {label}")

        # Assertions
        assert img.shape == (
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
        assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"
        assert (
            0 <= label.item() < len(encoder.id_to_class)
        ), "Label index out of bounds."
    print("Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n--- 3. Verifying Model ---")
    num_classes = len(encoder.id_to_class)
    print(f"Initializing model for {num_classes} classes...")

    # Initialize model (pretrained=False for speed in demo)
    model = HotelResNet(num_classes=num_classes, pretrained=False)
    model.eval()

    # Create dummy input batch (Batch Size 2)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        num_classes,
    ), f"Model output shape mismatch. Expected (2, {num_classes}), got {output.shape}"
    print("Model verification passed.")

    # ---------------------------------------------------------
    # 4. Run Training (Debug Mode)
    # ---------------------------------------------------------
    print("\n--- 4. Running Training (Fast Mode) ---")
    # Using debug=True subsets data to Config.DEBUG_SAMPLE_SIZE (1000)
    # Using epochs=1 to ensure quick completion
    best_map5 = run_training(
        debug=True, epochs=1, batch_size=16, load_cached_data=False
    )

    print(f"Training completed. Best MAP@5: {best_map5}")

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"
    print("Checkpoint verified.")

    # ---------------------------------------------------------
    # 5. Run Inference (Debug Mode)
    # ---------------------------------------------------------
    print("\n--- 5. Running Inference (Fast Mode) ---")
    generate_submission(
        checkpoint_path=Config.MODEL_CHECKPOINT,
        output_file=Config.SUBMISSION_FILE,
        batch_size=16,
        debug=True,
        load_cached_data=False,
    )

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    # Validate content format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {sub_df.shape}")
    print("Head:")
    print(sub_df.head(2))

    required_cols = ["image", "hotel_id"]
    assert all(
        col in sub_df.columns for col in required_cols
    ), f"Submission missing required columns. Found: {sub_df.columns}"

    # Check format of hotel_id (should be space-delimited string)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "hotel_id column should contain strings."
    assert len(sample_pred.split()) <= 5, "Predictions should not exceed 5 items."

    print("Inference verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
