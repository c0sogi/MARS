import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import CACHE_DIR, DEVICE, WORKING_DIR
from library.utils import set_seed, get_logger
from library.data_loader import process_and_cache_data, get_data_loaders
from library.model import MSSCWBN, DualPooling, CBAM, predict
from library.train_eval import train_fold

# Initialize Logger
logger = get_logger("demo_script")


def run_demo():
    print("Starting Demonstration Script...")

    # 1. Setup and Reproducibility
    set_seed(42)
    print("Random seed set to 42.")

    # 2. Data Loading and Verification
    print("\n[Step 1] Verifying Data Processing...")
    # Load processed data (this triggers caching logic)
    data = process_and_cache_data(load_cached_data=True)

    # Verify Data Dictionary Structure
    required_keys = [
        "train_images",
        "train_angles",
        "train_labels",
        "train_ids",
        "test_images",
        "test_angles",
        "test_ids",
    ]
    for key in required_keys:
        assert key in data, f"Missing key in processed data: {key}"

    # Verify Shapes
    # Images: (N, 3, 75, 75)
    assert data["train_images"].ndim == 4
    assert data["train_images"].shape[1] == 3
    assert data["train_images"].shape[2] == 75
    assert data["train_images"].shape[3] == 75

    # Angles and Labels: (N,)
    assert data["train_angles"].ndim == 1
    assert data["train_labels"].ndim == 1
    assert len(data["train_images"]) == len(data["train_labels"])

    print(f"Data shapes verified. Train samples: {len(data['train_images'])}")

    # 3. DataLoader Verification
    print("\n[Step 2] Verifying DataLoaders...")
    batch_size = 8
    train_loader, val_loader, test_loader = get_data_loaders(
        load_cached_data=True, batch_size=batch_size
    )

    # Fetch one batch from training loader
    images, angles, labels = next(iter(train_loader))

    # Verify Batch Tensors
    assert images.shape == (
        batch_size,
        3,
        75,
        75,
    ), f"Incorrect image batch shape: {images.shape}"
    assert angles.shape == (
        batch_size,
        1,
    ), f"Incorrect angle batch shape: {angles.shape}"
    assert labels.shape == (
        batch_size,
        1,
    ), f"Incorrect label batch shape: {labels.shape}"
    assert images.dtype == torch.float32
    print("DataLoader batch shapes verified.")

    # 4. Model Component Verification
    print("\n[Step 3] Verifying Model Components...")

    # Test DualPooling
    # Input: (B, C, H, W) -> Output: (B, 2C, H/2, W/2)
    dummy_input = torch.randn(2, 64, 32, 32)
    pool = DualPooling(kernel_size=2, stride=2)
    pool_out = pool(dummy_input)
    assert pool_out.shape == (
        2,
        128,
        16,
        16,
    ), f"DualPooling output mismatch: {pool_out.shape}"
    print("DualPooling logic verified.")

    # Test CBAM
    # Input: (B, C, H, W) -> Output: (B, C, H, W)
    cbam = CBAM(planes=64)
    cbam_out = cbam(dummy_input)
    assert (
        cbam_out.shape == dummy_input.shape
    ), f"CBAM output mismatch: {cbam_out.shape}"
    print("CBAM logic verified.")

    # 5. Full Model Forward Pass
    print("\n[Step 4] Verifying Full Model Forward Pass...")
    model = MSSCWBN().to(DEVICE)

    # Move dummy batch to device
    images_dev = images.to(DEVICE)
    angles_dev = angles.to(DEVICE)

    # Forward pass
    output = model(images_dev, angles_dev)

    # Output should be (Batch, 1) logits
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass successful.")

    # 6. Training Loop Demonstration (Short Run)
    print("\n[Step 5] Demonstrating Training Loop (1 Fold, 2 Epochs)...")

    # We use the train_fold function from library.train_eval
    # We limit epochs to 2 for speed
    fold_idx = 0
    trained_model = train_fold(
        fold_idx=fold_idx,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        patience=1,
        device=DEVICE,
    )

    # Verify artifact creation
    expected_model_path = os.path.join(
        CACHED_DIR := "./working/idea_37", f"model_fold_{fold_idx}.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model artifact not found at {expected_model_path}"
    print(f"Training complete. Model saved to {expected_model_path}")

    # 7. Prediction and Submission Generation
    print("\n[Step 6] Generating Predictions...")

    # Predict on test set using the trained model
    test_preds = predict(trained_model, test_loader, device=DEVICE)

    assert len(test_preds) == len(data["test_ids"]), "Prediction count mismatch"
    assert np.all(
        (test_preds >= 0) & (test_preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Create submission dataframe
    submission_df = pd.DataFrame({"id": data["test_ids"], "is_iceberg": test_preds})

    # Save to working directory for verification
    demo_sub_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_sub_path, index=False)
    print(f"Predictions generated. Sample:\n{submission_df.head()}")
    print(f"Demo submission saved to {demo_sub_path}")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
