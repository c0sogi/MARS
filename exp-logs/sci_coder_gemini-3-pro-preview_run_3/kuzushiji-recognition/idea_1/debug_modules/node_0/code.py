import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library import config, utils, dataset, model, loss, trainer, inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Configuration Overrides for Speed
    # We modify the config module directly to affect downstream classes
    print("\n[1] Configuring environment for rapid demonstration...")

    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4
    config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 images for training/val/test
    config.NUM_WORKERS = 0  # Use main thread for simplicity in demo
    config.IMG_SIZE = 512  # Reduce image size for faster compute in demo

    # Ensure reproducibility
    utils.seed_everything(config.SEED)
    utils.setup_directories()

    print(f"    Batch Size: {config.BATCH_SIZE}")
    print(f"    Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"    Debug Mode: Enabled (Sample Size: {config.DEBUG_SAMPLE_SIZE})")

    # 2. Dataset Verification
    print("\n[2] Verifying Dataset Logic...")

    # Initialize dataset in debug mode
    train_ds = dataset.KuzushijiDataset(
        split="train", load_cached_data=False, debug=True
    )

    # Check length
    assert (
        len(train_ds) == config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length mismatch. Expected {config.DEBUG_SAMPLE_SIZE}, got {len(train_ds)}"

    # Fetch one item
    sample = train_ds[0]

    # Verify keys
    expected_keys = {
        "image",
        "image_id",
        "hm",
        "ind",
        "reg",
        "cls_ids",
        "reg_mask",
        "center",
        "scale",
    }
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset item. Found: {sample.keys()}"

    # Verify Shapes
    # Image should be (3, H, W)
    img_shape = sample["image"].shape
    assert img_shape == (
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Incorrect image shape: {img_shape}"

    # Heatmap should be (1, H/4, W/4)
    hm_shape = sample["hm"].shape
    expected_hm_shape = (1, config.IMG_SIZE // 4, config.IMG_SIZE // 4)
    assert (
        hm_shape == expected_hm_shape
    ), f"Incorrect heatmap shape: {hm_shape}. Expected: {expected_hm_shape}"

    print("    Dataset shapes and content verified successfully.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")

    device = config.DEVICE
    net = model.SparseCenterNet().to(device)

    # Create dummy input batch
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)

    # Forward pass
    hm_out, reg_out, emb_out = net(dummy_input)

    # Verify output shapes
    # Heatmap: (B, 1, H/4, W/4)
    assert hm_out.shape == (
        2,
        1,
        config.IMG_SIZE // 4,
        config.IMG_SIZE // 4,
    ), "Heatmap output shape mismatch"
    # Regression: (B, 2, H/4, W/4)
    assert reg_out.shape == (
        2,
        2,
        config.IMG_SIZE // 4,
        config.IMG_SIZE // 4,
    ), "Regression output shape mismatch"
    # Embedding: (B, 64, H/4, W/4)
    assert emb_out.shape == (
        2,
        64,
        config.IMG_SIZE // 4,
        config.IMG_SIZE // 4,
    ), "Embedding output shape mismatch"

    print("    Model forward pass successful. Output shapes correct.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Calculation...")

    # Instantiate Loss
    criterion = loss.SparseCenterNetLoss(net.classifier).to(device)

    # Create dummy targets simulating a batch from the dataset
    # We need to collate the sample we fetched earlier to make a batch of size 1
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = [v]  # List for non-tensors like image_id

    # Forward pass on this real sample (but using the un-trained model)
    outputs = net(batch["image"])

    # Compute loss
    total_loss, stats = criterion(outputs, batch)

    assert not torch.isnan(total_loss), "Loss returned NaN"
    assert total_loss.item() > 0, "Loss should be positive"
    assert (
        "loss_hm" in stats and "loss_reg" in stats and "loss_cls" in stats
    ), "Missing loss components in stats"

    print(f"    Loss calculation successful. Total Loss: {total_loss.item():.4f}")

    # 5. Training Loop Verification
    print("\n[5] Running Training Loop (Trainer)...")

    # Initialize Trainer with debug=True
    # This will use the config overrides we set earlier
    tm = trainer.Trainer(debug=True, load_cached_data=False)

    # Run fit for 1 epoch
    tm.fit(num_epochs=1)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    print("    Training loop completed. Checkpoint saved.")

    # 6. Inference Verification
    print("\n[6] Running Inference and Submission Generation...")

    # Run generation
    # This uses the checkpoint saved in the previous step
    inference.generate_submission(checkpoint_path=checkpoint_path, debug=True)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_FILE_PATH), "Submission file not created"

    # Verify content format
    df_sub = pd.read_csv(config.SUBMISSION_FILE_PATH)
    assert (
        "image_id" in df_sub.columns and "labels" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"    Submission generated at {config.SUBMISSION_FILE_PATH}")
    print(f"    Rows generated: {len(df_sub)}")

    print("\n=== Demonstration Complete: All Systems Operational ===")


if __name__ == "__main__":
    main()
