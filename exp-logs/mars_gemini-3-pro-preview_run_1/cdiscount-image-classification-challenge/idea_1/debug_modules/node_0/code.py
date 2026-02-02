import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import CdiscountDataset, collate_fn
from library.model import MultiViewResNet
from library.trainer import Trainer
from library.utils import load_checkpoint, get_transforms


def run_demo():
    # 1. Setup and Reproducibility
    print("==== Step 1: Setup ====")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Ensure metadata exists (sanity check)
    assert os.path.exists(Config.TRAIN_METADATA), "Train metadata not found"
    assert os.path.exists(Config.TRAIN_BSON), "Train BSON not found"

    # 2. Dataset and Data Loading Verification
    print("\n==== Step 2: Verifying Dataset & Collate Function ====")

    # Initialize dataset without transforms first to check raw tensor structure
    # We use the training metadata
    dataset = CdiscountDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        transform=get_transforms("train"),
        mode="train",
    )

    # Fetch a single sample
    # Expected output: images (N, 3, 180, 180), target (int), sample_id (int)
    images, target, sample_id = dataset[0]

    print(f"Sample 0 - Image Stack Shape: {images.shape}")
    print(f"Sample 0 - Target Category Index: {target}")
    print(f"Sample 0 - Sample ID: {sample_id}")

    # Assertions
    assert images.dim() == 4, "Images should be a 4D tensor (N, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert images.shape[2] == 180 and images.shape[3] == 180, "Images should be 180x180"
    assert isinstance(target, int), "Target should be an integer"

    # Test DataLoader with custom collate_fn
    # We use a small batch size
    batch_size = 4
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    # Get one batch
    batch_images, batch_indices, batch_targets, batch_ids = next(iter(loader))

    print(f"Batch - Flattened Images Shape: {batch_images.shape}")
    print(f"Batch - Indices Shape: {batch_indices.shape}")
    print(f"Batch - Targets Shape: {batch_targets.shape}")

    # Assertions for Batch
    # batch_images: (Sum of N_i, 3, 180, 180)
    # batch_indices: (Sum of N_i,)
    # batch_targets: (Batch_Size,)
    assert (
        batch_images.shape[0] == batch_indices.shape[0]
    ), "Image count and indices count mismatch"
    assert batch_targets.shape[0] == batch_size, f"Expected {batch_size} targets"
    assert batch_ids.shape[0] == batch_size, f"Expected {batch_size} sample IDs"

    # Verify that indices correspond to batch elements (0 to batch_size-1)
    unique_indices = torch.unique(batch_indices)
    assert unique_indices.max().item() < batch_size, "Batch indices exceed batch size"

    # 3. Model Architecture Verification
    print("\n==== Step 3: Verifying Model Architecture ====")

    model = MultiViewResNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    model.eval()

    # Move batch to device
    batch_images = batch_images.to(device)
    batch_indices = batch_indices.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(batch_images, batch_indices)

    print(f"Model Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected ({batch_size}, {Config.NUM_CLASSES}), got {logits.shape}"

    # 4. Training Loop Demonstration
    print("\n==== Step 4: Running Short Training Loop (Demo) ====")

    # Initialize Trainer
    trainer = Trainer()

    # Run fit with debug limits to ensure speed
    # We limit to 200 samples and 1 epoch
    debug_limit = 200
    print(f"Training on subset of {debug_limit} samples for 1 epoch...")

    trainer.fit(num_epochs=1, batch_size=16, debug_limit=debug_limit)

    # Verify checkpoint creation
    checkpoint_path = Config.MODEL_CHECKPOINT
    assert os.path.exists(checkpoint_path), "Model checkpoint was not created"
    print(f"Checkpoint verified at: {checkpoint_path}")

    # 5. Inference/Submission Logic Verification
    print("\n==== Step 5: Verifying Inference Logic ====")

    # Load the trained model from checkpoint
    inference_model = MultiViewResNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    epoch, best_acc = load_checkpoint(checkpoint_path, inference_model)
    inference_model = inference_model.to(device)
    inference_model.eval()

    print(f"Loaded model from Epoch {epoch}, Best Val Acc: {best_acc:.4f}")

    # Create Test Dataset (Subset for speed)
    test_dataset = CdiscountDataset(
        metadata_path=Config.TEST_METADATA,
        bson_path=Config.TEST_BSON,
        transform=get_transforms("test"),
        mode="test",
    )

    # Manually subset test dataset to first 50 samples for verification
    test_subset = torch.utils.data.Subset(test_dataset, range(50))

    test_loader = DataLoader(
        test_subset, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    predictions = []
    ids = []

    with torch.no_grad():
        for imgs, idxs, _, s_ids in test_loader:
            imgs = imgs.to(device)
            idxs = idxs.to(device)

            # Forward
            outputs = inference_model(imgs, idxs)
            _, preds = outputs.max(1)

            predictions.extend(preds.cpu().numpy())
            ids.extend(s_ids.numpy())

    print(f"Generated {len(predictions)} predictions.")

    # Verify output format
    assert len(predictions) == 50, "Prediction count mismatch"
    assert len(ids) == 50, "ID count mismatch"

    # Check if predictions are valid class indices
    assert (
        min(predictions) >= 0 and max(predictions) < Config.NUM_CLASSES
    ), "Predictions out of valid class range"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
