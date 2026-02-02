import os
import shutil
import numpy as np
import torch
import pandas as pd
import library.utils as utils
import library.layers as layers
from library.model import WideAntiAliasedCoordRes2NeXt
import library.dataset as dataset_lib
import library.trainer as trainer_lib
import library.inference as inference_lib


def main():
    # 1. Setup
    print("--- 1. Setup and Initialization ---")
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Set seed for reproducibility
    utils.set_seed(42)
    device = utils.get_device()
    print(f"Device: {device}")

    # 2. Verify Layers
    print("\n--- 2. Verifying Custom Layers ---")

    # Test BlurPool
    # Input: (Batch, Channels, Height, Width)
    x = torch.randn(2, 3, 32, 32)
    blur = layers.BlurPool(channels=3, stride=2)
    out = blur(x)
    print(f"BlurPool Output Shape: {out.shape}")
    # Stride 2 should halve dimensions: 32 -> 16
    assert out.shape == (2, 3, 16, 16), "BlurPool output shape mismatch"

    # Test GeM
    x = torch.randn(2, 64, 8, 8)
    gem = layers.GeM(p=3.0)
    out = gem(x)
    print(f"GeM Output Shape: {out.shape}")
    # GeM pools spatial dims to 1x1
    assert out.shape == (2, 64, 1, 1), "GeM output shape mismatch"

    # Test CoordinateAttention
    x = torch.randn(2, 32, 16, 16)
    ca = layers.CoordinateAttention(inp=32, reduction=8)
    out = ca(x)
    print(f"CoordinateAttention Output Shape: {out.shape}")
    assert out.shape == (2, 32, 16, 16), "CoordinateAttention output shape mismatch"

    # Test Res2NeXtBlock
    # In: 64, Out: 64*4=256 (expansion=4), Stride=1
    x = torch.randn(2, 64, 32, 32)
    block = layers.Res2NeXtBlock(
        in_planes=64, planes=64, stride=1, cardinality=4, scales=4
    )
    out = block(x)
    print(f"Res2NeXtBlock Output Shape: {out.shape}")
    assert out.shape == (2, 256, 32, 32), "Res2NeXtBlock output shape mismatch"

    # 3. Data Loading & Dataset Verification
    print("\n--- 3. Verifying Data Loading & Dataset ---")

    # Load data (this uses caching mechanism in library/dataset.py)
    # Note: This might take a few seconds if cache needs to be built
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids) = (
        dataset_lib.load_data()
    )

    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")

    assert len(train_imgs) == len(
        train_labels
    ), "Train images and labels count mismatch"
    assert train_imgs.dtype == np.uint8, "Images should be uint8"

    # Test Dataset Class
    ds = dataset_lib.CactusDataset(
        train_imgs[:10],
        train_labels[:10],
        transform=dataset_lib.get_transforms("train"),
    )
    sample_img, sample_label = ds[0]

    print(f"Dataset Sample Shape: {sample_img.shape}")
    print(f"Dataset Sample Label: {sample_label}")

    # Check tensor properties
    assert isinstance(sample_img, torch.Tensor), "Dataset should return tensors"
    assert sample_img.shape == (3, 32, 32), "Dataset image shape should be (3, 32, 32)"
    # Check normalization (should be 0-1 range roughly, definitely float)
    assert sample_img.dtype == torch.float32
    assert (
        sample_img.max() <= 1.0 and sample_img.min() >= 0.0
    ), "Image data not normalized to [0, 1]"

    # 4. Model Verification
    print("\n--- 4. Verifying Model Architecture ---")
    model = WideAntiAliasedCoordRes2NeXt(num_classes=1).to(device)

    # Pass a dummy batch
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Model output shape should be (Batch, 1)"

    # 5. Training Demonstration
    print("\n--- 5. Running Training Cycle (Demo) ---")

    # Subset data for speed
    subset_size = 256
    train_subset = (train_imgs[:subset_size], train_labels[:subset_size])
    val_subset = (val_imgs[:subset_size], val_labels[:subset_size])

    # Run training for 1 epoch
    # We use a temporary directory for checkpoints
    ckpt_dir = os.path.join(DEMO_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_model_path = trainer_lib.run_training_cycle(
        seed=42,
        train_data=train_subset,
        val_data=val_subset,
        working_dir=ckpt_dir,
        batch_size=32,
        epochs=1,  # Only 1 epoch for demo
        lr=1e-3,
        patience=1,
    )

    print(f"Training finished. Best model saved to: {best_model_path}")
    assert os.path.exists(best_model_path), "Model checkpoint file was not created"

    # 6. Inference Demonstration
    print("\n--- 6. Running Inference (Demo) ---")

    # Subset test data for speed
    test_imgs_subset = test_imgs[:100]
    test_ids_subset = test_ids[:100]

    submission_dir = os.path.join(DEMO_DIR, "submission")

    inference_lib.generate_submission(
        test_imgs=test_imgs_subset,
        test_ids=test_ids_subset,
        model_paths=[best_model_path],
        output_dir=submission_dir,
        batch_size=32,
    )

    sub_file = os.path.join(submission_dir, "submission.csv")
    print(f"Inference finished. Submission saved to: {sub_file}")

    assert os.path.exists(sub_file), "Submission file was not created"

    # Verify submission content
    df_sub = pd.read_csv(sub_file)
    print(f"Submission shape: {df_sub.shape}")
    assert df_sub.shape == (100, 2), "Submission shape mismatch"
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Submission columns mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
