import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import load_megadetector_data, get_crop_coordinates
from library.dataset import CameraTrapDataset
from library.model import FrozenResNetClassifier
from library.train import run_training, generate_predictions


def main():
    print("=== Starting Library Demonstration and Verification ===")

    # 1. Setup and Configuration
    Config.set_seed(Config.SEED)
    Config.make_dirs()

    print(f"Device: {Config.DEVICE}")
    print(f"Input Dir: {Config.INPUT_DIR}")
    print(f"Metadata Dir: {Config.METADATA_DIR}")

    # Ensure metadata exists (sanity check based on task description)
    assert os.path.exists(Config.TRAIN_CSV), "Train metadata missing"
    assert os.path.exists(Config.TEST_CSV), "Test metadata missing"

    # 2. Verify Utils: MegaDetector Loading and Coordinate Calculation
    print("\n--- Verifying Utils ---")

    # Test loading MegaDetector data
    # We use load_cached_data=False first to force parsing logic, then True to test cache
    detections = load_megadetector_data(load_cached_data=False)
    assert isinstance(
        detections, dict
    ), "load_megadetector_data should return a dictionary"
    print(f"Loaded detections for {len(detections)} images.")

    # Test coordinate calculation
    # Case A: No detection -> Full image
    img_w, img_h = 1000, 800
    coords = get_crop_coordinates(img_w, img_h, detection_info=None)
    assert coords == (0, 0, 1000, 800), f"Expected full crop, got {coords}"

    # Case B: Valid detection
    # bbox format in JSON is [x, y, w, h] normalized
    # Let's say x=0.1, y=0.1, w=0.5, h=0.5 -> absolute: 100, 80, 500, 400 -> x2=600, y2=480
    det_info = {"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 0.99}
    coords = get_crop_coordinates(img_w, img_h, det_info, conf_threshold=0.5)
    # Expected: x_min=100, y_min=80, x_max=600, y_max=480
    assert coords == (100, 80, 600, 480), f"Expected (100, 80, 600, 480), got {coords}"

    # Case C: Low confidence -> Full image
    det_info_low = {"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 0.1}
    coords = get_crop_coordinates(img_w, img_h, det_info_low, conf_threshold=0.5)
    assert coords == (
        0,
        0,
        1000,
        800,
    ), f"Expected full crop due to low conf, got {coords}"
    print("Utils verification passed.")

    # 3. Verify Dataset
    print("\n--- Verifying Dataset ---")

    # Use a small sample size for speed
    sample_size = 16

    # Train Dataset
    train_ds = CameraTrapDataset(
        split="train", sample_size=sample_size, load_cached_data=True
    )
    assert (
        len(train_ds) == sample_size
    ), f"Expected {sample_size} samples, got {len(train_ds)}"

    img_tensor, label = train_ds[0]

    # Check Tensor shape: (3, 224, 224)
    assert isinstance(img_tensor, torch.Tensor), "Output image should be a Tensor"
    assert img_tensor.shape == (
        3,
        224,
        224,
    ), f"Expected shape (3, 224, 224), got {img_tensor.shape}"
    assert isinstance(label, int), "Train label should be an integer"

    # Test Dataset
    test_ds = CameraTrapDataset(
        split="test", sample_size=sample_size, load_cached_data=True
    )
    img_tensor_test, img_id = test_ds[0]

    assert isinstance(img_id, str), "Test dataset should return image ID as string"
    assert img_tensor_test.shape == (3, 224, 224), "Test image shape mismatch"
    print("Dataset verification passed.")

    # 4. Verify Model
    print("\n--- Verifying Model ---")

    model = FrozenResNetClassifier(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # Check freezing logic
    # Backbone parameters should have requires_grad = False
    # FC parameters should have requires_grad = True
    backbone_frozen = all(
        not p.requires_grad for p in model.backbone.conv1.parameters()
    )
    head_trainable = all(p.requires_grad for p in model.backbone.fc.parameters())

    assert backbone_frozen, "Backbone layers should be frozen"
    assert head_trainable, "Head layer should be trainable"

    # Check Forward Pass
    dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
    output = model(dummy_input)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model verification passed.")

    # 5. Verify Training Loop
    print("\n--- Verifying Training Loop ---")

    # Run training for 1 epoch with small batch size and sample size
    # This tests the integration of Dataset, Model, and Training logic
    trained_model = run_training(
        sample_size=32,
        epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        patience=1,
        load_cached_data=True,
    )

    assert isinstance(
        trained_model, torch.nn.Module
    ), "run_training should return a model"

    # Check if best model was saved
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model_custom.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved"
    print("Training loop verification passed.")

    # 6. Verify Inference
    print("\n--- Verifying Inference ---")

    # Generate predictions on a small test set
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create a DataLoader for inference verification
    test_loader = torch.utils.data.DataLoader(
        test_ds,  # defined above with sample_size=16
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    generate_predictions(
        trained_model, test_loader, torch.device(Config.DEVICE), submission_path
    )

    assert os.path.exists(submission_path), "Submission file not created"

    # Validate submission format
    df_sub = pd.read_csv(submission_path)
    assert (
        "Id" in df_sub.columns and "Category" in df_sub.columns
    ), "Submission missing required columns"
    assert (
        len(df_sub) == sample_size
    ), f"Submission length mismatch. Expected {sample_size}, got {len(df_sub)}"
    assert pd.api.types.is_integer_dtype(
        df_sub["Category"]
    ), "Category column should be integers"

    print("Inference verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
