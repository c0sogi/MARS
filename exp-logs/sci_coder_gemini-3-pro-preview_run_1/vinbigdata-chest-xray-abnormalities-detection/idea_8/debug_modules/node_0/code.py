import os
import torch
import pandas as pd
import numpy as np
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DetModel
from library.loss import CenterNetLoss
from library.engine import Engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_metadata(n_samples=50):
    """
    Creates a small subset of the metadata to speed up dataset initialization
    and training for demonstration purposes.
    """
    print(f"Creating metadata subsets with {n_samples} samples...")

    # Define new paths in working directory
    subset_dir = os.path.join(Config.WORK_DIR, "demo_meta")
    os.makedirs(subset_dir, exist_ok=True)

    train_sub_path = os.path.join(subset_dir, "train_meta.csv")
    val_sub_path = os.path.join(subset_dir, "val_meta.csv")
    test_sub_path = os.path.join(subset_dir, "test_meta.csv")

    # Read original metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Sample and Save
    # We group by image_id to ensure we don't split objects of the same image
    train_imgs = df_train["image_id"].unique()[:n_samples]
    val_imgs = df_val["image_id"].unique()[:n_samples]
    test_imgs = df_test["image_id"].unique()[:n_samples]

    df_train_sub = df_train[df_train["image_id"].isin(train_imgs)]
    df_val_sub = df_val[df_val["image_id"].isin(val_imgs)]
    df_test_sub = df_test[df_test["image_id"].isin(test_imgs)]

    df_train_sub.to_csv(train_sub_path, index=False)
    df_val_sub.to_csv(val_sub_path, index=False)
    df_test_sub.to_csv(test_sub_path, index=False)

    return train_sub_path, val_sub_path, test_sub_path


def verify_model_architecture(device):
    """
    Verifies that the model outputs tensors of the correct shape.
    """
    print("Verifying model architecture...")
    model = DetModel(Config).to(device)
    model.eval()

    # Create dummy input: (Batch, Channels, Height, Width)
    # Config.IMAGE_SIZE is 640
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    # Expected Output Stride is 4
    expected_h = Config.IMAGE_SIZE // 4
    expected_w = Config.IMAGE_SIZE // 4

    # Check Heatmap Shape: (B, NumClasses, H/4, W/4)
    hm_shape = outputs["hm"].shape
    assert hm_shape == (
        2,
        Config.NUM_CLASSES,
        expected_h,
        expected_w,
    ), f"Heatmap shape mismatch. Expected {(2, Config.NUM_CLASSES, expected_h, expected_w)}, got {hm_shape}"

    # Check Regression Shape: (B, 2, H/4, W/4)
    reg_shape = outputs["reg"].shape
    assert reg_shape == (
        2,
        2,
        expected_h,
        expected_w,
    ), f"Regression offset shape mismatch. Got {reg_shape}"

    # Check WH Shape: (B, 2, H/4, W/4)
    wh_shape = outputs["wh"].shape
    assert wh_shape == (
        2,
        2,
        expected_h,
        expected_w,
    ), f"WH shape mismatch. Got {wh_shape}"

    # Check Global Class Shape: (B, 1)
    global_shape = outputs["global_cls"].shape
    assert global_shape == (
        2,
        1,
    ), f"Global classification shape mismatch. Got {global_shape}"

    print("Model architecture verification passed.")
    return model


def verify_loss_function(device):
    """
    Verifies that the loss function runs and returns a scalar.
    """
    print("Verifying loss function...")
    criterion = CenterNetLoss()

    batch_size = 2
    out_size = Config.IMAGE_SIZE // 4
    num_classes = Config.NUM_CLASSES
    max_objs = 100

    # Create dummy predictions
    outputs = {
        "hm": torch.randn(batch_size, num_classes, out_size, out_size).to(device),
        "wh": torch.randn(batch_size, 2, out_size, out_size).to(device),
        "reg": torch.randn(batch_size, 2, out_size, out_size).to(device),
        "global_cls": torch.randn(batch_size, 1).to(device),
    }

    # Create dummy targets
    batch = {
        "hm": torch.zeros(batch_size, num_classes, out_size, out_size).to(device),
        "wh": torch.zeros(batch_size, max_objs, 2).to(device),
        "reg": torch.zeros(batch_size, max_objs, 2).to(device),
        "ind": torch.zeros(batch_size, max_objs, dtype=torch.int64).to(device),
        "reg_mask": torch.zeros(batch_size, max_objs).to(device),
        "global_label": torch.zeros(batch_size, 1).to(device),
    }

    # Forward pass loss
    loss, stats = criterion(outputs, batch)

    assert torch.is_tensor(loss), "Loss is not a tensor"
    assert loss.dim() == 0, "Loss is not a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print("Loss function verification passed.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Override Config for Demo
    # We use a specific demo directory to avoid clutter
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")

    # Reduce compute requirements
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Ensure directories exist
    Config.setup()

    # 3. Create and Link Subset Data
    train_sub, val_sub, test_sub = create_subset_metadata(n_samples=20)
    Config.TRAIN_META_PATH = train_sub
    Config.VAL_META_PATH = val_sub
    Config.TEST_META_PATH = test_sub

    # 4. Verify Components
    model = verify_model_architecture(device)
    verify_loss_function(device)

    # 5. Initialize Engine Components
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Get DataLoaders (will use the subset paths defined in Config)
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    engine = Engine(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=None,  # Skip scheduler for short demo
    )

    # 6. Run Training Pipeline
    # debug=True limits the loop to a few batches per epoch
    print("Starting Engine execution...")
    engine.run(train_loader, val_loader, test_loader, epochs=1, debug=True)

    # 7. Verify Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated successfully with {len(df_sub)} rows.")
        # Basic format check
        assert "ID" in df_sub.columns or "image_id" in df_sub.columns
        assert "PredictionString" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("Demo completed successfully.")
