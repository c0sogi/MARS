import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_class_weights, calculate_roc_auc
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet34
from library.engine import train_fn, eval_fn, inference_fn


def main():
    print("Starting Apple Disease Detection Library Demonstration...")

    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    # Override Config for a fast demonstration run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_execution"
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "output")
    Config.MODELS_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup()

    # Set seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Preparation (Subsampling)
    # ==========================================
    print("\n[Data] Loading and subsampling metadata...")

    # Load original metadata
    train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create small subsets for demonstration (ensure we don't exceed available data)
    train_subset = train_full.head(16).copy()
    val_subset = val_full.head(8).copy()
    test_subset = test_full.head(8).copy()

    # Save temporary subset metadata for weight calculation test
    temp_train_meta_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    train_subset.to_csv(temp_train_meta_path, index=False)

    print(
        f"Train subset: {len(train_subset)}, Val subset: {len(val_subset)}, Test subset: {len(test_subset)}"
    )

    # ==========================================
    # 3. Verify Class Weights
    # ==========================================
    print("\n[Utils] Verifying Class Weights calculation...")
    # Force calculation from the new subset file, ignoring cache
    weights = get_class_weights(temp_train_meta_path, load_cached_data=False)

    print(f"Class Weights: {weights}")
    assert isinstance(weights, torch.Tensor), "Weights should be a torch.Tensor"
    assert (
        weights.shape[0] == Config.NUM_CLASSES
    ), f"Weights shape mismatch. Expected {Config.NUM_CLASSES}, got {weights.shape[0]}"
    assert not torch.isnan(weights).any(), "Weights contain NaN values"

    # ==========================================
    # 4. Verify Dataset and Transforms
    # ==========================================
    print("\n[Dataset] Verifying AppleDataset and Transforms...")

    # Instantiate dataset
    train_ds = AppleDataset(
        train_subset, transforms=get_transforms("train"), output_label=True
    )

    # Fetch one item
    image, label = train_ds[0]

    print(f"Image Shape: {image.shape}")
    print(f"Label: {label}")

    # Assertions
    assert image.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {image.shape}"
    assert label.shape == (
        Config.NUM_CLASSES,
    ), f"Label shape mismatch. Expected ({Config.NUM_CLASSES},), got {label.shape}"
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    # ==========================================
    # 5. Verify Model Architecture
    # ==========================================
    print("\n[Model] Verifying AppleResNet34 architecture...")

    model = AppleResNet34(
        num_classes=Config.NUM_CLASSES, pretrained=False
    )  # False for speed
    model.to(device)

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")

    assert dummy_output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {dummy_output.shape}"

    # ==========================================
    # 6. Training Loop Demonstration
    # ==========================================
    print("\n[Engine] Demonstrating Training Loop (1 Epoch)...")

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )

    val_ds = AppleDataset(
        val_subset, transforms=get_transforms("valid"), output_label=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train Function
    train_loss, train_auc = train_fn(model, train_loader, criterion, optimizer, device)
    print(f"Train Result -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert isinstance(train_auc, float), "Train AUC should be a float"

    # ==========================================
    # 7. Validation Loop Demonstration
    # ==========================================
    print("\n[Engine] Demonstrating Validation Loop...")

    val_loss, val_auc, val_preds, val_targets = eval_fn(
        model, val_loader, criterion, device, use_tta=False
    )
    print(f"Val Result   -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert val_preds.shape == (
        len(val_subset),
        Config.NUM_CLASSES,
    ), "Validation predictions shape mismatch"
    assert val_targets.shape == (
        len(val_subset),
        Config.NUM_CLASSES,
    ), "Validation targets shape mismatch"

    # ==========================================
    # 8. Inference Demonstration (with TTA)
    # ==========================================
    print("\n[Engine] Demonstrating Inference with TTA...")

    test_ds = AppleDataset(
        test_subset, transforms=get_transforms("test"), output_label=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_preds = inference_fn(model, test_loader, device, use_tta=True)

    print(f"Test Predictions Shape: {test_preds.shape}")

    assert test_preds.shape == (
        len(test_subset),
        Config.NUM_CLASSES,
    ), f"Test predictions shape mismatch. Expected ({len(test_subset)}, {Config.NUM_CLASSES}), got {test_preds.shape}"

    # Check probabilities sum to approx 1 (Softmax applied in inference_fn)
    row_sums = test_preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # ==========================================
    # 9. Generate Submission File
    # ==========================================
    print("\n[Submission] Generating sample submission file...")

    submission_df = pd.DataFrame(test_preds, columns=Config.CLASSES)
    submission_df.insert(0, "image_id", test_subset["image_id"].values)

    print("Sample Submission Head:")
    print(submission_df.head())

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
