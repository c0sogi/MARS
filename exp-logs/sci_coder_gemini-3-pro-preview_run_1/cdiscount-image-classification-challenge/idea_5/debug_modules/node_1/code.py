import os
import sys
import torch
import pandas as pd
import numpy as np

# Import provided libraries
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.trainer as trainer_lib


def main():
    print("==== Starting Demonstration Script ====")

    # Ensure reproducibility
    trainer_lib.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # ==========================================
    # 1. Demonstrate HierarchyMapper
    # ==========================================
    print("\n[1] Testing HierarchyMapper...")
    # Force recompute to verify logic (load_cached_data=False)
    mapper = utils.HierarchyMapper(load_cached_data=False)

    stats = mapper.get_num_classes()
    print(
        f"   Mapped Classes: L1={stats['num_classes_l1']}, L2={stats['num_classes_l2']}, L3={stats['num_classes_l3']}"
    )

    # Validation
    assert stats["num_classes_l3"] > 0, "Level 3 classes should be > 0"
    assert not mapper.mappings_df.empty, "Mappings DataFrame should not be empty"

    # Check lookup for a specific category
    if not mapper.mappings_df.empty:
        sample_cat_id = mapper.mappings_df["category_id"].iloc[0]
        labels = mapper.get_labels(sample_cat_id)

        assert labels is not None, "Lookup failed for valid category_id"
        assert (
            "l1" in labels and "l2" in labels and "l3" in labels
        ), "Labels dictionary missing keys"
    print("   HierarchyMapper validation successful.")

    # ==========================================
    # 2. Demonstrate Dataset & DataLoader
    # ==========================================
    print("\n[2] Testing Dataset and DataLoader...")

    # Use debug mode with a small subset for speed
    subset_size = 32
    batch_size = 8

    train_loader, val_loader, test_loader, _ = dataset.get_dataloaders(
        debug=True, subset_size=subset_size, batch_size=batch_size, num_workers=2
    )

    # Fetch one batch
    images, l1_targets, l2_targets, l3_targets = next(iter(train_loader))

    print(f"   Batch Shapes - Images: {images.shape}, L3 Targets: {l3_targets.shape}")

    # Validation
    # Expected shape: (Batch, 4, 3, H, W) -> 4 is the fixed number of views per product
    assert (
        images.dim() == 5
    ), "Images tensor should be 5D (Batch, Views, Channels, H, W)"
    assert images.size(1) == 4, "Should have 4 views per product"
    assert images.size(2) == 3, "Should have 3 channels (RGB)"
    assert (
        images.size(3) == config.IMG_SIZE and images.size(4) == config.IMG_SIZE
    ), f"Image size should be {config.IMG_SIZE}"
    assert l3_targets.size(0) == batch_size, "Target batch size mismatch"
    print("   Dataset validation successful.")

    # ==========================================
    # 3. Demonstrate Model Architecture
    # ==========================================
    print("\n[3] Testing Model Architecture...")

    # Instantiate model with stats from mapper
    # We use pretrained=False here to speed up the demo instantiation
    model = model_lib.DeepSupervisedResNet50(
        num_classes_l1=stats["num_classes_l1"],
        num_classes_l2=stats["num_classes_l2"],
        num_classes_l3=stats["num_classes_l3"],
        pretrained=False,
    )
    model.to(device)

    # Forward pass with the batch fetched earlier
    images = images.to(device)

    with torch.no_grad():
        logits_l1, logits_l2, logits_l3 = model(images)

    print(
        f"   Output Shapes - L1: {logits_l1.shape}, L2: {logits_l2.shape}, L3: {logits_l3.shape}"
    )

    # Validation
    assert logits_l1.shape == (batch_size, stats["num_classes_l1"])
    assert logits_l2.shape == (batch_size, stats["num_classes_l2"])
    assert logits_l3.shape == (batch_size, stats["num_classes_l3"])
    print("   Model architecture validation successful.")

    # ==========================================
    # 4. Demonstrate Full Training Loop
    # ==========================================
    print("\n[4] Running Integration Test (Training Loop)...")

    # We use the provided run_training function which encapsulates the Trainer.
    # We run for 1 epoch on a slightly larger subset (64 samples).
    # Note: run_training forces pretrained=True, so this might download weights if not cached.

    trainer = trainer_lib.run_training(
        debug=True, subset_size=64, epochs=1, batch_size=16
    )

    # Check if model checkpoint was saved
    expected_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        expected_path
    ), f"Model checkpoint not found at {expected_path}"
    print(f"   Training finished. Checkpoint saved at: {expected_path}")

    # ==========================================
    # 5. Demonstrate Inference & Submission
    # ==========================================
    print("\n[5] Generating Sample Submission...")

    # Load best model weights
    # Re-instantiate model with pretrained=True to match training config
    model = model_lib.DeepSupervisedResNet50(
        num_classes_l1=stats["num_classes_l1"],
        num_classes_l2=stats["num_classes_l2"],
        num_classes_l3=stats["num_classes_l3"],
        pretrained=True,
    )
    model.to(device)
    model.load_state_dict(torch.load(expected_path, map_location=device))
    model.eval()

    predictions = []
    sample_ids = []

    # Create an inverse mapping: l3_idx -> category_id
    l3_to_cat_id = dict(
        zip(mapper.mappings_df["l3_idx"], mapper.mappings_df["category_id"])
    )

    print("   Running inference on test subset...")
    # Iterate over test loader (using the one created in step 2 for this demo)
    for batch_imgs, batch_ids in test_loader:
        batch_imgs = batch_imgs.to(device)

        with torch.no_grad():
            # Model returns (l1, l2, l3)
            _, _, logits_l3 = model(batch_imgs)

            # Get predictions
            preds = torch.argmax(logits_l3, dim=1).cpu().numpy()

            # Store
            predictions.extend(preds)
            sample_ids.extend(batch_ids.numpy())

    # Map predictions back to original category_ids
    final_category_ids = [l3_to_cat_id.get(p, 0) for p in predictions]

    # Create submission DataFrame
    submission_df = pd.DataFrame({"_id": sample_ids, "category_id": final_category_ids})

    print(f"   Generated {len(submission_df)} predictions.")
    print("   Sample rows:")
    print(submission_df.head())

    # Validate format
    assert "_id" in submission_df.columns
    assert "category_id" in submission_df.columns
    assert len(submission_df) > 0

    # Save to working directory
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"   Submission saved to {submission_path}")

    print("\n==== Demonstration Complete ====")


if __name__ == "__main__":
    main()
