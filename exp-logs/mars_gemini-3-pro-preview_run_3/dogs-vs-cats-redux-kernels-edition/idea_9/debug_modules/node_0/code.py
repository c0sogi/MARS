import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.models import get_model
from library.engine import fit_model, predict


def demo_implementation():
    print("Starting implementation demo...")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # Define parameters for a quick run
    # Using a very small subset to ensure execution within seconds
    DEBUG_SUBSET = 32
    BATCH_SIZE = 8
    RESOLUTION = 224
    MODEL_NAME = "resnet101"

    # 2. Demonstrate Data Loading
    print("\n[Demo] Data Loading")
    # We use debug_subset to load only a few images
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        num_workers=2,
        load_cached_data=False,  # Force processing from csv
        debug_subset=DEBUG_SUBSET,
    )

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"Train batch shape: Images {images.shape}, Labels {labels.shape}")

        assert images.shape == (
            BATCH_SIZE,
            3,
            RESOLUTION,
            RESOLUTION,
        ), "Train image batch shape mismatch"
        assert labels.shape == (BATCH_SIZE,), "Train label batch shape mismatch"
        assert labels.dtype == torch.float32, "Train label dtype mismatch"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Test Loader (returns image, id)
    try:
        test_images, test_ids = next(iter(test_loader))
        print(f"Test batch shape: Images {test_images.shape}, IDs {test_ids.shape}")

        assert test_images.shape == (
            BATCH_SIZE,
            3,
            RESOLUTION,
            RESOLUTION,
        ), "Test image batch shape mismatch"
        # IDs are usually ints or longs, checking length matches batch
        assert len(test_ids) == BATCH_SIZE, "Test ID batch size mismatch"
    except StopIteration:
        raise AssertionError("Test loader is empty!")

    print("Data loading logic verified.")

    # 3. Demonstrate Model Instantiation
    print(f"\n[Demo] Model Instantiation ({MODEL_NAME})")
    # Instantiating with pretrained=False for speed in this specific check.
    # Note: fit_model below will use pretrained=True as hardcoded in library.engine.
    model = get_model(model_name=MODEL_NAME, num_classes=1, pretrained=False)
    model = model.to(device)
    model.eval()

    # Verify Architecture (Custom Head Check)
    # ResNet101 in this library uses a custom GeM head
    if hasattr(model, "pool"):
        print(f"Pooling layer: {model.pool}")
        assert "GeM" in str(model.pool), "Expected GeM pooling for ResNet101"
    else:
        raise AssertionError(
            "Expected 'pool' attribute (GeM) in ResNet101 model wrapper"
        )

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, RESOLUTION, RESOLUTION).to(device)
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model logic verified.")

    # 4. Demonstrate Training Engine
    print("\n[Demo] Training Engine (fit_model)")
    # This function encapsulates the training loop.
    # We run for 1 epoch on the debug subset.
    # Note: This will attempt to download weights if not cached, as library.engine forces pretrained=True.
    trained_model = fit_model(
        model_name=MODEL_NAME,
        resolution=RESOLUTION,
        epochs=1,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        patience=1,
        num_workers=2,
        debug_subset=DEBUG_SUBSET,
    )

    assert isinstance(
        trained_model, torch.nn.Module
    ), "fit_model returned invalid object"
    print("Training engine execution verified.")

    # 5. Demonstrate Prediction
    print("\n[Demo] Prediction")
    # Using the trained model to predict on the test loader (subset)
    ids, probs = predict(trained_model, test_loader, device, use_tta=True)

    print(f"Number of predictions: {len(ids)}")
    if len(probs) > 0:
        print(
            f"Probabilities stats: Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}"
        )

    # Validations
    assert len(ids) == len(probs), "IDs and Probabilities count mismatch"
    expected_count = len(test_loader.dataset)
    assert (
        len(ids) == expected_count
    ), f"Expected {expected_count} predictions, got {len(ids)}"

    assert (probs >= 0.0).all() and (
        probs <= 1.0
    ).all(), "Probabilities out of [0,1] range"

    print("Prediction logic verified.")
    print("\nSUCCESS: All demonstrations passed.")


if __name__ == "__main__":
    demo_implementation()
