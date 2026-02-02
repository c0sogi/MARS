import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.engine as engine


def setup_demo_paths():
    """
    Redirects library paths to a demo directory to ensure isolation.
    """
    demo_dir = "./working/demo_execution"
    cache_dir = os.path.join(demo_dir, "cache")
    checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    submission_dir = os.path.join(demo_dir, "submission")

    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Monkey-patch config paths
    config.WORKING_DIR = demo_dir
    config.CHECKPOINT_DIR = checkpoint_dir
    config.SUBMISSION_DIR = submission_dir
    config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Monkey-patch data paths (since they are imported into data.py namespace)
    data.TRAIN_CACHE_PATH = os.path.join(cache_dir, "train_processed.npz")
    data.TEST_CACHE_PATH = os.path.join(cache_dir, "test_processed.npz")

    print(f"Demo working directory set to: {demo_dir}")
    return submission_dir


def demonstrate_pipeline():
    # 1. Setup
    utils.set_seed(42)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\n--- Data Loading ---")
    # Using fold 0, variant 'A' (Mean channel)
    train_loader, val_loader, test_loader = data.create_dataloaders(
        fold=0, n_splits=5, variant="A", load_cached_data=True
    )

    # Verification: Check DataLoaders
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"
    assert len(test_loader) > 0, "Test loader is empty"

    # Verification: Check Batch Shapes
    # Fetch one batch
    images, angles, targets, ids = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Expected: (B, 3, 224, 224) for images (upsampled), (B,) for angles
    assert images.shape[1:] == (3, 224, 224), f"Incorrect image shape: {images.shape}"
    assert angles.dim() == 1, f"Incorrect angle dimension: {angles.dim()}"

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    net = model.IcebergResNet(dropout_rate=0.5)
    net.to(device)

    # Verification: Forward Pass
    with torch.no_grad():
        dummy_images = images.to(device)
        dummy_angles = angles.to(device)
        logits = net(dummy_images, dummy_angles)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (images.size(0), 1), "Output shape mismatch"

    # 4. Training Loop (1 Epoch)
    print("\n--- Training (1 Epoch) ---")
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # Train for one epoch
    train_loss = engine.train_one_epoch(
        model=net,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        label_smoothing=0.05,
    )
    print(f"Training Loss: {train_loss:.4f}")
    assert np.isfinite(train_loss), "Training loss is not finite"

    # 5. Validation with TTA
    print("\n--- Validation (TTA) ---")
    val_loss = engine.validate_tta(net, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    assert np.isfinite(val_loss), "Validation loss is not finite"

    # 6. SWA Phase
    print("\n--- SWA Phase ---")
    # Run SWA for just 1 epoch to demonstrate functionality
    swa_model = engine.run_swa_phase(
        model=net,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        swa_epochs=1,
        swa_lr=5e-5,
        label_smoothing=0.05,
    )

    # Verification: Check if it's an AveragedModel
    assert isinstance(
        swa_model, AveragedModel
    ), "Returned model is not an AveragedModel"
    print("SWA Phase completed successfully.")

    # 7. Prediction / Inference
    print("\n--- Inference ---")
    # We use the SWA model for prediction
    # predict_ensemble expects a list of models
    pred_ids, pred_probs = engine.predict_ensemble([swa_model], test_loader, device)

    print(f"Number of predictions: {len(pred_probs)}")

    # Verification: Check counts
    # We need to know the total test size. We can get it from the loader dataset.
    total_test_samples = len(test_loader.dataset)
    assert (
        len(pred_probs) == total_test_samples
    ), f"Prediction count {len(pred_probs)} mismatch with dataset size {total_test_samples}"

    # 8. Submission File Generation
    print("\n--- Generating Submission ---")
    df_sub = pd.DataFrame({"id": pred_ids, "is_iceberg": pred_probs})

    save_path = config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to: {save_path}")

    # Verify file exists
    assert os.path.exists(save_path), "Submission file was not created"

    # Verify content format
    df_check = pd.read_csv(save_path)
    print("First 5 rows of submission:")
    print(df_check.head())
    assert list(df_check.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect columns in submission"


if __name__ == "__main__":
    try:
        setup_demo_paths()
        demonstrate_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Re-raise to ensure the task fails if the code fails
        raise e
