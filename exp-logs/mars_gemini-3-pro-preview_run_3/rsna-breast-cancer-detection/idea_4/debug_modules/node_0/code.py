import sys
import os
import numpy as np
import torch
import cv2
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, probabilistic_f1, apply_analytical_correction
from library.data_processing import preprocess_image, crop_breast_roi, load_dicom
from library.dataset import get_dataloaders, prepare_bag_dataframe, BreastBagDataset
from library.model import BreastMILModel
from library.train_eval import run_training


def configure_for_demo():
    """
    Overrides Config parameters to ensure the script runs quickly and purely for demonstration/verification.
    """
    print("Configuring environment for fast demonstration...")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 bags per split

    # Reduce image size drastically for speed
    Config.IMG_HEIGHT = 128
    Config.IMG_WIDTH = 64

    # Training hyperparameters for quick convergence/exit
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.PATIENCE = 1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)


def test_utils():
    print("\n=== Testing Utils ===")

    # 1. Test Probabilistic F1
    # Perfect match
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    pf1_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(pf1_perfect, 1.0), f"Expected pF1 1.0, got {pf1_perfect}"

    # Worst match
    y_pred_worst = np.array([0.0, 1.0, 0.0, 1.0])
    pf1_worst = probabilistic_f1(y_true, y_pred_worst)
    assert np.isclose(pf1_worst, 0.0), f"Expected pF1 0.0, got {pf1_worst}"

    print("Probabilistic F1 logic verified.")

    # 2. Test Analytical Correction
    # If train_prev == test_prev, shift should be 0
    logits = torch.tensor([0.0])  # Sigmoid(0) = 0.5
    corrected = apply_analytical_correction(
        logits, train_prevalence=0.2, test_prevalence=0.2
    )
    assert torch.isclose(
        corrected, torch.tensor([0.5])
    ), "Correction should be identity when prevalences match"

    # If test prevalence is lower, probability should decrease
    corrected_lower = apply_analytical_correction(
        logits, train_prevalence=0.5, test_prevalence=0.02
    )
    assert (
        corrected_lower < 0.5
    ), "Probability should decrease for lower test prevalence"

    print("Analytical correction logic verified.")


def test_data_processing():
    print("\n=== Testing Data Processing ===")

    # Create a synthetic image (uint8)
    # Background black (0), Breast tissue white (255)
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

    # 1. Test Crop
    cropped = crop_breast_roi(img)
    # The crop should be roughly 100x100 (the rectangle)
    h, w = cropped.shape
    assert 90 <= h <= 110 and 90 <= w <= 110, f"Crop failed, shape: {cropped.shape}"
    print("ROI Cropping verified.")

    # 2. Test Preprocess
    # Resize to Config dimensions (128, 64)
    processed = preprocess_image(
        cropped, target_height=Config.IMG_HEIGHT, target_width=Config.IMG_WIDTH
    )

    assert processed.shape == (
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
        3,
    ), f"Wrong shape: {processed.shape}"
    assert processed.dtype == np.float32, "Wrong dtype"
    assert processed.max() <= 1.0 and processed.min() >= 0.0, "Normalization failed"
    print("Image preprocessing verified.")


def test_dataset_pipeline():
    print("\n=== Testing Dataset & DataLoader ===")

    # We rely on get_dataloaders which uses the Config we modified
    # Force load_cached_data=False to ensure we process the CSVs and test that logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train Loader Length: {len(train_loader)}")

    # Fetch one batch
    images, labels, ids = next(iter(train_loader))

    # Verify Batch Structure
    # images is a list of tensors (Bag size varies, but collate returns list)
    assert isinstance(images, list), "Images should be a list (MIL bags)"
    assert (
        len(images) == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {len(images)}"

    # Verify Tensor Shape inside list
    # Shape: (N_views, 3, H, W)
    sample_bag = images[0]
    assert sample_bag.ndim == 4
    assert sample_bag.shape[1] == 3
    assert sample_bag.shape[2] == Config.IMG_HEIGHT
    assert sample_bag.shape[3] == Config.IMG_WIDTH

    # Verify Labels
    assert labels.shape == (Config.BATCH_SIZE,), "Labels shape mismatch"
    assert labels.dtype == torch.float32

    print("Dataset and DataLoader pipeline verified.")
    return images  # Return for model testing


def test_model(sample_images):
    print("\n=== Testing Model Architecture ===")

    # Instantiate model
    # We use pretrained=False to speed up initialization (avoid download) for this logic check
    model = BreastMILModel(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Move inputs to device
    images_device = [img.to(Config.DEVICE) for img in sample_images]

    # Forward pass
    with torch.no_grad():
        logits = model(images_device)

    # Check output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {logits.shape}"

    print("Model forward pass verified.")


def test_full_training_loop():
    print("\n=== Testing Full Training Loop (Integration) ===")

    # This calls the provided train_eval.py function
    # It will use the Config overrides (DEBUG=True, Epochs=1)
    # This verifies that the whole pipeline connects correctly

    try:
        run_training()
        print("Training loop completed successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e


if __name__ == "__main__":
    # 1. Setup
    configure_for_demo()

    # 2. Unit Tests
    test_utils()
    test_data_processing()

    # 3. Component Tests
    sample_batch = test_dataset_pipeline()
    test_model(sample_batch)

    # 4. Integration Test
    test_full_training_loop()

    print("\nAll demonstrations and verifications passed.")
