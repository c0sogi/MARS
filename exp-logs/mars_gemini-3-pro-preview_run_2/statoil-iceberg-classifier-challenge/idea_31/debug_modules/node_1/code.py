import os
import torch
import pandas as pd
import numpy as np
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def run_demo():
    print("Starting Demo Script...")

    # ==========================================
    # 1. CONFIGURATION OVERRIDES FOR DEMO
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")
    # Override config to run quickly
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 16
    config.NUM_FOLDS = 2  # We will only run fold 0

    # Use a specific directory for demo artifacts to avoid conflicts
    DEMO_DIR = "./working/demo_artifacts"
    os.makedirs(DEMO_DIR, exist_ok=True)
    config.MODEL_CHECKPOINT_DIR = DEMO_DIR
    config.PROCESSED_DATA_PATH = os.path.join(
        DEMO_DIR, "cache", "processed_data_debug.npz"
    )

    utils.seed_everything(config.SEED)
    print(f"Configured: Epochs={config.NUM_EPOCHS}, Batch Size={config.BATCH_SIZE}")

    # ==========================================
    # 2. DATA PROCESSING VERIFICATION
    # ==========================================
    print("\n[2] Verifying Data Processing...")
    # Force reload to verify processing logic
    data_dict = utils.load_and_process_data(load_cached_data=False)

    # Assertions to verify data structure
    required_keys = ["train_images", "train_targets", "train_inc_angles", "test_images"]
    for key in required_keys:
        assert key in data_dict, f"Missing key in processed data: {key}"

    # Verify Shapes
    # Images should be (N, 3, 75, 75)
    assert data_dict["train_images"].ndim == 4, "Train images should be 4D"
    assert data_dict["train_images"].shape[1] == 3, "Images should have 3 channels"
    assert data_dict["train_images"].shape[2] == 75, "Height should be 75"
    assert data_dict["train_images"].shape[3] == 75, "Width should be 75"

    print(f"Data Loaded Successfully. Train shape: {data_dict['train_images'].shape}")

    # ==========================================
    # 3. DATASET & DATALOADER VERIFICATION
    # ==========================================
    print("\n[3] Verifying Dataset and DataLoader...")

    # Test Dataset __getitem__
    ds = data.IcebergDataset(
        images=data_dict["train_images"][:10],
        inc_angles=data_dict["train_inc_angles"][:10],
        targets=data_dict["train_targets"][:10],
        transform=data.get_transforms(mode="train"),
        stats=data_dict["stats"],
    )

    img, angle, target = ds[0]

    # Verify individual item
    assert isinstance(img, torch.Tensor), "Dataset should return image tensor"
    assert img.shape == (3, 75, 75), f"Unexpected image shape: {img.shape}"
    assert isinstance(angle, torch.Tensor), "Dataset should return angle tensor"
    assert isinstance(target, torch.Tensor), "Dataset should return target tensor"

    # Test DataLoader
    train_loader, val_loader = data.get_dataloaders(fold_idx=0, load_cached_data=True)

    # Fetch one batch
    images_batch, angles_batch, targets_batch = next(iter(train_loader))

    assert images_batch.shape[0] == config.BATCH_SIZE, "Batch size mismatch"
    assert images_batch.shape[1] == 3, "Channel mismatch in batch"
    print("DataLoader verified successfully.")

    # ==========================================
    # 4. MODEL ARCHITECTURE VERIFICATION
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    net = model.RDP_WBN()
    net.to(config.DEVICE)
    net.eval()

    # Create dummy inputs
    dummy_img = torch.randn(4, 3, 75, 75).to(config.DEVICE)
    dummy_angle = torch.randn(4).to(config.DEVICE)  # 1D tensor for angles

    with torch.no_grad():
        output = net(dummy_img, dummy_angle)

    # Verify output
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Model forward pass successful.")

    # ==========================================
    # 5. TRAINING LOOP EXECUTION
    # ==========================================
    print("\n[5] Executing Training Loop (Fold 0)...")

    # Run training for Fold 0
    # This uses the overridden config.NUM_EPOCHS=2
    best_val_loss = train.run_fold(fold_idx=0)

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")

    expected_model_path = os.path.join(DEMO_DIR, "model_fold_0.pth")
    assert os.path.exists(expected_model_path), "Model checkpoint was not saved."

    # ==========================================
    # 6. INFERENCE & SUBMISSION GENERATION
    # ==========================================
    print("\n[6] Running Inference and Generating Submission...")

    # Load Test Loader
    test_loader = data.get_test_dataloader(load_cached_data=True)

    # Load Model
    net = model.RDP_WBN()
    net.load_state_dict(torch.load(expected_model_path, map_location=config.DEVICE))
    net.to(config.DEVICE)
    net.eval()

    predictions = []
    ids = []

    # We need the IDs corresponding to the test loader.
    # The loader doesn't return IDs, so we get them from the data dictionary directly
    # Note: The test_loader is sequential and shuffle=False, so order is preserved.
    test_ids = data_dict["test_ids"]

    with torch.no_grad():
        for i, (images, angles) in enumerate(test_loader):
            images = images.to(config.DEVICE)
            angles = angles.to(config.DEVICE)

            # Forward
            logits = net(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)

    # Truncate IDs to match predictions (in case drop_last was somehow active, though it isn't by default)
    # or ensure lengths match
    assert len(predictions) == len(
        test_ids
    ), f"Mismatch: {len(predictions)} preds vs {len(test_ids)} ids"

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save
    submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")
    df_sub.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(df_sub.head())

    # ==========================================
    # 7. FINAL VALIDATION
    # ==========================================
    print("\n[7] Final Validation...")
    assert os.path.exists(submission_path), "Submission file not found"

    df_check = pd.read_csv(submission_path)
    assert list(df_check.columns) == ["id", "is_iceberg"], "Submission columns mismatch"
    assert len(df_check) == len(test_ids), "Submission row count mismatch"
    assert df_check["is_iceberg"].min() >= 0.0, "Probabilities must be >= 0"
    assert df_check["is_iceberg"].max() <= 1.0, "Probabilities must be <= 1"

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
