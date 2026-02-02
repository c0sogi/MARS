import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import warnings

# Import library modules
# We import config first to patch it before other modules might use the values
import library.config
import library.train
import library.predict
import library.dataset
import library.model
import library.utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Patch Configuration for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Patching configuration for rapid execution...")

    # Reduce epochs and folds to minimal values for demonstration
    library.config.EPOCHS = 1
    library.config.N_FOLDS = 2
    library.config.BATCH_SIZE = 16

    # Propagate changes to dependent modules that imported constants
    library.train.EPOCHS = library.config.EPOCHS
    library.train.N_FOLDS = library.config.N_FOLDS
    library.train.BATCH_SIZE = library.config.BATCH_SIZE

    library.predict.N_FOLDS = library.config.N_FOLDS
    library.predict.BATCH_SIZE = library.config.BATCH_SIZE

    # Set seed for reproducibility
    library.utils.seed_everything(library.config.SEED)
    device = library.utils.get_device()
    print(f"    Device: {device}")
    print("    Configuration patched successfully.")

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n[2] Testing Data Loading (Train Mode)...")

    # Load training data
    # This triggers processing of JSON -> Images, scaling, and caching
    images, angles, labels = library.dataset.load_data(
        mode="train", load_cached_data=True
    )

    # Validations
    print(f"    Loaded {len(images)} images.")

    # Check shapes
    assert len(images) == len(angles) == len(labels), "Data lengths mismatch"
    assert images.ndim == 4, f"Images should be 4D (N, H, W, C), got {images.shape}"
    assert images.shape[1:] == (
        224,
        224,
        3,
    ), f"Expected (224, 224, 3), got {images.shape[1:]}"
    assert angles.ndim == 1, "Angles should be 1D"

    # Check Normalization
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images not normalized to [0, 1]"
    assert not np.isnan(angles).any(), "Angles contain NaNs"

    print("    Data integrity checks passed.")

    # ------------------------------------------------------------------------
    # 3. Dataset and DataLoader
    # ------------------------------------------------------------------------
    print("\n[3] Testing IcebergDataset and DataLoader...")

    # Create dataset with training transforms
    train_transform = library.dataset.get_transforms(mode="train")
    dataset = library.dataset.IcebergDataset(
        images, angles, labels, transform=train_transform
    )

    # Test __getitem__
    sample_img, sample_angle, sample_label = dataset[0]

    # Verify Tensor conversion
    assert isinstance(sample_img, torch.Tensor), "Image is not a Tensor"
    assert isinstance(sample_angle, torch.Tensor), "Angle is not a Tensor"
    assert isinstance(sample_label, torch.Tensor), "Label is not a Tensor"

    # Verify Channel-First format (C, H, W) for PyTorch
    assert sample_img.shape == (
        3,
        224,
        224,
    ), f"Expected (3, 224, 224), got {sample_img.shape}"

    print("    Dataset __getitem__ checks passed.")

    # ------------------------------------------------------------------------
    # 4. Model Initialization
    # ------------------------------------------------------------------------
    print("\n[4] Testing Model Initialization and Forward Pass...")

    model = library.model.IcebergEfficientNet()
    model.to(device)
    model.eval()

    # Create dummy batch
    dummy_img = torch.randn(4, 3, 224, 224).to(device)
    dummy_angle = torch.tensor([0.5, 0.6, 0.7, 0.8]).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    # Verify output shape (Batch, 1)
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"

    print("    Model forward pass successful.")

    # ------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # ------------------------------------------------------------------------
    print("\n[5] Simulating Training Loop (1 Epoch, Subset)...")

    # Use a small subset for speed
    subset_indices = list(range(32))
    subset_dataset = Subset(dataset, subset_indices)

    loader = DataLoader(
        subset_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run train_one_epoch
    train_loss = library.train.train_one_epoch(
        model, loader, criterion, optimizer, device, label_smoothing=0.05
    )

    print(f"    Training Step Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"

    # Run validate
    val_loss = library.train.validate(model, loader, criterion, device)
    print(f"    Validation Step Loss: {val_loss:.4f}")

    # Save a dummy checkpoint to test saving mechanism
    dummy_ckpt_path = os.path.join(library.config.WORKING_DIR, "demo_checkpoint.pth")
    library.utils.save_checkpoint(model, optimizer, 1, val_loss, dummy_ckpt_path)
    assert os.path.exists(dummy_ckpt_path), "Checkpoint file was not created"

    print("    Training functions verified.")

    # ------------------------------------------------------------------------
    # 6. Inference Simulation (TTA)
    # ------------------------------------------------------------------------
    print("\n[6] Testing Inference with TTA...")

    # Use the same subset loader for inference test
    # Note: TTA requires images and angles, loader yields labels too but predict_with_tta handles unpacking if dataset yields 2 items?
    # Checking library.predict.predict_with_tta:
    # "for images, angles in loader:" -> This expects the loader to yield exactly 2 items.
    # However, our subset_dataset (IcebergDataset) yields 3 items (img, ang, lbl) when labels are present.
    # We need to create a dataset without labels for the predict function to work as written,
    # or the predict function must handle the 3rd item.
    # Looking at library.dataset.IcebergDataset: "if self.labels is not None: return image, angle, label"
    # Looking at library.predict.predict_with_tta: "for images, angles in loader:" -> This will crash if loader yields 3 items.

    # Let's create a test-mode dataset (no labels) for this verification
    test_subset_images = images[:16]
    test_subset_angles = angles[:16]
    test_dataset_demo = library.dataset.IcebergDataset(
        test_subset_images,
        test_subset_angles,
        labels=None,
        transform=library.dataset.get_transforms(mode="valid"),
    )

    test_loader_demo = DataLoader(test_dataset_demo, batch_size=4, shuffle=False)

    # Run prediction
    preds = library.predict.predict_with_tta(model, test_loader_demo, device)

    assert preds.shape == (
        16,
        1,
    ), f"Expected predictions shape (16, 1), got {preds.shape}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions out of probability range [0, 1]"

    print("    Inference with TTA verified.")

    # ------------------------------------------------------------------------
    # 7. Mocking Full Pipeline (Submission)
    # ------------------------------------------------------------------------
    print("\n[7] Verifying Submission Generation Logic...")

    # We won't run the full generate_submission() because it loads the full test set and runs N folds.
    # Instead, we verify the output path and logic.

    submission_path = library.config.SUBMISSION_PATH
    print(f"    Target submission path: {submission_path}")

    # Manually create a dummy submission to verify file writing permissions
    dummy_ids = [f"id_{i}" for i in range(5)]
    dummy_preds = np.random.rand(5)
    df_sub = pd.DataFrame({"id": dummy_ids, "is_iceberg": dummy_preds})
    df_sub.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file not created"
    print("    Submission file creation verified.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
