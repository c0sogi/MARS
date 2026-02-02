import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dicom_loader import read_dicom_image
from library.dataset import VinBigDataset
from library.model import SpatiallyAwareCenterNet
from library.loss import CenterNetLoss
from library.train import train_one_epoch, validate
from library.inference import predict, post_process


def create_mini_metadata():
    """
    Creates a small subset of the metadata for rapid demonstration.
    """
    print("Creating mini-datasets for demonstration...")

    # Create working directory for metadata
    os.makedirs("./working/demo_meta", exist_ok=True)

    # 1. Train Subset (5 images)
    df_train = pd.read_csv(Config.TRAIN_META)
    train_ids = df_train["image_id"].unique()[:5]
    df_train_mini = df_train[df_train["image_id"].isin(train_ids)].copy()
    mini_train_path = "./working/demo_meta/train_meta.csv"
    df_train_mini.to_csv(mini_train_path, index=False)

    # 2. Val Subset (5 images)
    df_val = pd.read_csv(Config.VAL_META)
    val_ids = df_val["image_id"].unique()[:5]
    df_val_mini = df_val[df_val["image_id"].isin(val_ids)].copy()
    mini_val_path = "./working/demo_meta/val_meta.csv"
    df_val_mini.to_csv(mini_val_path, index=False)

    # 3. Test Subset (5 images)
    df_test = pd.read_csv(Config.TEST_META)
    test_ids = df_test["image_id"].unique()[:5]
    df_test_mini = df_test[df_test["image_id"].isin(test_ids)].copy()
    mini_test_path = "./working/demo_meta/test_meta.csv"
    df_test_mini.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def configure_demo_settings(train_path, val_path, test_path):
    """
    Overrides Config attributes for the demo run.
    """
    Config.TRAIN_META = train_path
    Config.VAL_META = val_path
    Config.TEST_META = test_path

    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Speed optimization
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.DEBUG = True

    # Re-run setup to create new dirs
    Config.setup()


def verify_dicom_loading(df_path):
    """
    Verifies that DICOM images can be read and processed.
    """
    print("\n--- Verifying DICOM Loader ---")
    df = pd.read_csv(df_path)

    # Pick the first image
    rel_path = df.iloc[0]["file_path"]
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    print(f"Reading: {full_path}")
    img, h, w = read_dicom_image(full_path)

    assert isinstance(img, np.ndarray), "Image should be a numpy array"
    assert len(img.shape) == 3, "Image should be 3-channel (H, W, 3)"
    assert img.dtype == np.uint8, "Image should be uint8"
    assert h > 0 and w > 0, "Dimensions should be positive"

    print(f"Success. Shape: {img.shape}, Orig Dims: {h}x{w}")


def verify_dataset_and_loader(csv_path):
    """
    Verifies the Dataset class and DataLoader.
    """
    print("\n--- Verifying Dataset & DataLoader ---")
    dataset = VinBigDataset(csv_path=csv_path, mode="train", load_cached_data=False)

    # Check length
    assert len(dataset) > 0, "Dataset should not be empty"

    # Check item structure
    sample = dataset[0]
    required_keys = [
        "image",
        "target_heatmap",
        "target_size",
        "target_offset",
        "target_mask",
        "global_label",
        "original_dims",
        "image_id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Check tensor shapes
    img_tensor = sample["image"]
    assert img_tensor.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image tensor shape: {img_tensor.shape}"

    # Check DataLoader
    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))

    print(f"Batch Image Shape: {batch['image'].shape}")
    print(f"Batch Heatmap Shape: {batch['target_heatmap'].shape}")

    return loader


def verify_model_and_loss(loader):
    """
    Verifies Model forward pass and Loss calculation.
    """
    print("\n--- Verifying Model & Loss ---")
    device = Config.DEVICE

    # Instantiate Model
    model = SpatiallyAwareCenterNet(pretrained=False).to(
        device
    )  # No pretrained weights for speed

    # Get a batch
    batch = next(iter(loader))

    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Forward Pass
    outputs = model(batch["image"])

    # Check Output Shapes
    # Output stride is 4, so 640 -> 160
    feat_size = Config.IMG_SIZE // 4
    assert outputs["hm"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        feat_size,
        feat_size,
    )
    assert outputs["wh"].shape == (Config.BATCH_SIZE, 2, feat_size, feat_size)
    assert outputs["reg"].shape == (Config.BATCH_SIZE, 2, feat_size, feat_size)
    assert outputs["global"].shape == (Config.BATCH_SIZE, 1)

    print("Forward pass successful.")

    # Loss Calculation
    criterion = CenterNetLoss()
    loss, stats = criterion(outputs, batch)

    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Loss Stats: {stats}")

    return model, criterion


def run_demo_training(model, train_loader, val_loader, criterion):
    """
    Runs a short training loop.
    """
    print("\n--- Running Demo Training (1 Epoch) ---")
    device = Config.DEVICE
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    print(f"Train Metrics: {train_metrics}")

    # Validate
    val_metrics = validate(model, val_loader, criterion, device)
    print(f"Val Metrics: {val_metrics}")

    # Save checkpoint manually for inference test
    torch.save(
        model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    )
    print("Saved demo checkpoint.")


def run_demo_inference():
    """
    Runs inference using the trained model on the mini test set.
    """
    print("\n--- Running Demo Inference ---")

    # Load Test Data
    test_dataset = VinBigDataset(
        csv_path=Config.TEST_META, mode="test", load_cached_data=False
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=0)

    # Load Model
    device = Config.DEVICE
    model = SpatiallyAwareCenterNet(pretrained=False).to(device)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Predict
    raw_results = predict(model, test_loader, device)

    assert len(raw_results) == len(test_dataset), "Results count mismatch"

    # Post-process
    df_sub = post_process(raw_results)

    print("Inference successful. Sample submission:")
    print(df_sub.head())

    # Verify format
    assert "image_id" in df_sub.columns
    assert "PredictionString" in df_sub.columns
    assert len(df_sub) == len(test_dataset)


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Create Mini Data
    mini_train, mini_val, mini_test = create_mini_metadata()

    # 3. Configure
    configure_demo_settings(mini_train, mini_val, mini_test)

    # 4. Verify Components
    verify_dicom_loading(mini_train)
    train_loader = verify_dataset_and_loader(mini_train)

    # Create val loader for training loop
    val_dataset = VinBigDataset(csv_path=mini_val, mode="val", load_cached_data=False)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    model, criterion = verify_model_and_loss(train_loader)

    # 5. Run Training
    run_demo_training(model, train_loader, val_loader, criterion)

    # 6. Run Inference
    run_demo_inference()

    print("\nAll demonstration steps completed successfully.")
