import os
import sys
import torch
import numpy as np
import pandas as pd
import time

# Import from the provided library files
from library.config import (
    OUTPUT_DIR,
    DEVICE,
    IMG_SIZE,
    NUM_CLASSES,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)
from library.utils import seed_everything, RocAucMeter
from library.layers import ECA, BlurPool, GeM
from library.model import CustomWideResNeXtECA
from library.dataset import get_loaders, CactusDataset
from library.trainer import run_training


def verify_layers():
    """
    Verifies the input/output shapes and basic execution of custom layers.
    """
    print("\n--- Verifying Custom Layers ---")

    batch_size = 4
    channels = 64
    height, width = 32, 32
    dummy_input = torch.randn(batch_size, channels, height, width)

    # 1. Test ECA (Efficient Channel Attention)
    eca = ECA(channels, gamma=2, b=1)
    out_eca = eca(dummy_input)
    assert (
        out_eca.shape == dummy_input.shape
    ), f"ECA output shape mismatch. Expected {dummy_input.shape}, got {out_eca.shape}"
    print("ECA Layer: Verified.")

    # 2. Test BlurPool
    # BlurPool usually preserves channels, might change spatial dims based on stride
    blur = BlurPool(channels, stride=1)
    out_blur = blur(dummy_input)
    assert (
        out_blur.shape == dummy_input.shape
    ), f"BlurPool (stride=1) output shape mismatch. Expected {dummy_input.shape}, got {out_blur.shape}"

    blur_strided = BlurPool(channels, stride=2)
    out_blur_strided = blur_strided(dummy_input)
    expected_shape = (batch_size, channels, height // 2, width // 2)
    assert (
        out_blur_strided.shape == expected_shape
    ), f"BlurPool (stride=2) output shape mismatch. Expected {expected_shape}, got {out_blur_strided.shape}"
    print("BlurPool Layer: Verified.")

    # 3. Test GeM (Generalized Mean Pooling)
    # GeM flattens spatial dimensions: (B, C, H, W) -> (B, C)
    gem = GeM(p=3.0)
    out_gem = gem(dummy_input)
    expected_gem_shape = (batch_size, channels)
    assert (
        out_gem.shape == expected_gem_shape
    ), f"GeM output shape mismatch. Expected {expected_gem_shape}, got {out_gem.shape}"
    print("GeM Layer: Verified.")


def verify_model():
    """
    Verifies the full model architecture.
    """
    print("\n--- Verifying Model Architecture ---")

    model = CustomWideResNeXtECA()
    model.to(DEVICE)
    model.eval()

    batch_size = 2
    # Input image: (B, 3, 32, 32)
    dummy_img = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    with torch.no_grad():
        logits = model(dummy_img)

    # Output should be (B, NUM_CLASSES) -> (B, 1)
    expected_shape = (batch_size, NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {logits.shape}"

    print(
        f"Model '{model.__class__.__name__}' instantiated and forward pass successful."
    )


def verify_data_loading():
    """
    Verifies the dataset and dataloader construction.
    Uses a small limit to speed up loading.
    """
    print("\n--- Verifying Data Loading ---")

    # Use a small limit to ensure speed
    limit = 50
    batch_size = 10

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=batch_size,
        load_cached_data=False,  # Force reload to test reading logic
        limit=limit,
    )

    # Check Train Loader
    images, labels, ids = next(iter(train_loader))

    # Images: (B, 3, 32, 32)
    assert (
        images.dim() == 4 and images.shape[1] == 3 and images.shape[2] == 32
    ), f"Train batch image shape incorrect: {images.shape}"

    # Labels: (B,) - float32
    assert (
        labels.dim() == 1 and labels.shape[0] == batch_size
    ), f"Train batch label shape incorrect: {labels.shape}"

    print(
        f"DataLoaders created successfully. Batch size: {batch_size}. Subset limit: {limit}."
    )
    return train_loader, val_loader, test_loader


def verify_training_loop():
    """
    Runs a minimal training loop using the library's trainer.
    """
    print("\n--- Verifying Training Loop ---")

    # Configuration for a quick run
    seed = 42
    epochs = 1
    batch_size = 16
    limit = 100  # Train on just 100 samples

    # Run training
    # This function returns the best AUC
    best_auc = run_training(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        weight_decay=1e-4,
        load_cached_data=False,  # Avoid using old cache for this demo
        limit=limit,
        patience=1,
    )

    assert isinstance(best_auc, float), "run_training should return a float (AUC)."
    assert 0.0 <= best_auc <= 1.0, f"AUC score {best_auc} is out of bounds [0, 1]."

    print(f"Training run complete. Best AUC: {best_auc}")

    # Check if model file was created
    model_path = os.path.join(OUTPUT_DIR, f"model_seed_{seed}.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print("Model checkpoint verified.")

    return model_path


def verify_inference(model_path):
    """
    Demonstrates how to load a trained model and run inference on the test set.
    """
    print("\n--- Verifying Inference ---")

    # 1. Load Data
    limit = 50
    _, _, test_loader = get_loaders(batch_size=16, load_cached_data=True, limit=limit)

    # 2. Load Model
    model = CustomWideResNeXtECA()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Inference Loop
    all_probs = []
    all_ids = []

    print("Running inference on test subset...")
    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_ids.extend(ids)

    # 4. Verify Output
    assert len(all_probs) == len(all_ids), "Mismatch between predictions and IDs."
    assert len(all_probs) <= limit, "Inference processed more items than the limit."

    # Create submission DataFrame
    submission = pd.DataFrame({"id": all_ids, "has_cactus": all_probs})

    print("Sample Inference Results:")
    print(submission.head())

    assert not submission.isnull().values.any(), "Submission contains NaN values."
    print("Inference verification complete.")


def main():
    # Set global seed
    seed_everything(42)

    # Ensure output directory exists (handled by config, but explicit check)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Verify Layers
    verify_layers()

    # 2. Verify Model
    verify_model()

    # 3. Verify Data Loading
    # We call this to ensure datasets work, though trainer calls it internally too.
    verify_data_loading()

    # 4. Verify Training
    # This runs the full pipeline for 1 epoch on a subset
    model_path = verify_training_loop()

    # 5. Verify Inference
    # Uses the model trained in step 4
    verify_inference(model_path)

    print("\nAll verifications passed successfully.")


if __name__ == "__main__":
    main()
