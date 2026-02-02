import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model
import library.train as train_lib


def main():
    print("Starting demonstration script...")

    # 1. Setup & Reproducibility
    # ---------------------------------------------------------
    SEED = 42
    utils.set_seed(SEED)
    print(f"Random seed set to {SEED}.")

    # Define working directories (using the ones from config)
    working_dir = config.WORKING_DIR
    submission_dir = config.SUBMISSION_DIR
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Data Loading
    # ---------------------------------------------------------
    # Load data arrays (cached or fresh)
    print("Loading data arrays...")
    data = dataset.get_data_arrays(load_cached_data=True)

    # Extract arrays
    train_images_full = data["train_images"]
    train_labels_full = data["train_labels"]
    val_images_full = data["val_images"]
    val_labels_full = data["val_labels"]
    test_images_full = data["test_images"]
    test_ids_full = data["test_ids"]

    print(f"Original Train shape: {train_images_full.shape}")
    print(f"Original Test shape: {test_images_full.shape}")

    # Optimization for Speed: Use a small subset
    SUBSET_SIZE = 128
    VAL_SUBSET_SIZE = 64
    TEST_SUBSET_SIZE = 64

    train_images = train_images_full[:SUBSET_SIZE]
    train_labels = train_labels_full[:SUBSET_SIZE]
    val_images = val_images_full[:VAL_SUBSET_SIZE]
    val_labels = val_labels_full[:VAL_SUBSET_SIZE]
    test_images = test_images_full[:TEST_SUBSET_SIZE]
    test_ids = test_ids_full[:TEST_SUBSET_SIZE]

    print(f"Subset Train shape: {train_images.shape}")

    # 3. Dataset Verification
    # ---------------------------------------------------------
    print("Verifying Dataset class...")
    # Get transforms
    train_transform = dataset.get_transforms(mode="train")

    # Instantiate Dataset
    train_ds = dataset.CactusDataset(
        images=train_images, labels=train_labels, transform=train_transform
    )

    # Verify length
    assert len(train_ds) == SUBSET_SIZE, "Dataset length mismatch."

    # Verify item structure
    img_tensor, label_tensor = train_ds[0]

    # Check shape: Should be (C, H, W) -> (3, 32, 32)
    assert img_tensor.shape == (
        3,
        32,
        32,
    ), f"Incorrect image tensor shape: {img_tensor.shape}"
    # Check type
    assert isinstance(img_tensor, torch.Tensor), "Image is not a tensor."
    assert isinstance(label_tensor, torch.Tensor), "Label is not a tensor."
    # Check normalization (approximate check for 0-1 range)
    assert (
        img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    ), "Image tensor not normalized to [0, 1]."

    print("Dataset verification passed.")

    # 4. Model Verification
    # ---------------------------------------------------------
    print("Verifying Model architecture...")
    device = config.DEVICE

    # Instantiate Model
    net = model.WideSEResNeXt(
        channels=config.CHANNELS, cardinality=config.CARDINALITY
    ).to(device)

    # Create dummy batch
    dummy_batch_size = 4
    dummy_input = torch.randn(dummy_batch_size, 3, 32, 32).to(device)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    # Check output shape: (Batch, 1)
    assert output.shape == (
        dummy_batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(dummy_batch_size, 1)}, got {output.shape}"

    print("Model verification passed.")

    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("Demonstrating training loop (1 epoch)...")

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution
    )

    val_ds = dataset.CactusDataset(
        images=val_images,
        labels=val_labels,
        transform=dataset.get_transforms(mode="val"),
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Optimizer & Loss
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Run 1 Epoch of Training
    train_loss, train_auc = train_lib.train_one_epoch(
        net, train_loader, optimizer, criterion, device
    )

    # Run Validation
    val_loss, val_auc = train_lib.validate(net, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Assertions to ensure training happened
    assert not np.isnan(train_loss), "Training loss is NaN."
    assert 0.0 <= train_auc <= 1.0, "Train AUC out of range."
    assert not np.isnan(val_loss), "Validation loss is NaN."

    print("Training loop demonstration passed.")

    # 6. Inference Demonstration (TTA)
    # ---------------------------------------------------------
    print("Demonstrating TTA Inference...")

    # Use the trained model (even if just 1 epoch) to predict
    # predict_tta expects numpy array of images
    preds = train_lib.predict_tta(net, test_images, device, batch_size=32)

    # Verify predictions
    assert len(preds) == TEST_SUBSET_SIZE, "Prediction count mismatch."
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]."
    assert preds.shape == (
        TEST_SUBSET_SIZE,
    ), f"Prediction shape mismatch: {preds.shape}"

    print(f"Generated {len(preds)} predictions. Mean prob: {preds.mean():.4f}")
    print("Inference demonstration passed.")

    # 7. Generate Submission File
    # ---------------------------------------------------------
    print("Generating sample submission file...")

    # Create DataFrame
    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

    # Save
    save_path = os.path.join(submission_dir, "submission_demo.csv")
    sub_df.to_csv(save_path, index=False)

    assert os.path.exists(save_path), "Submission file was not created."
    print(f"Submission saved to {save_path}")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
