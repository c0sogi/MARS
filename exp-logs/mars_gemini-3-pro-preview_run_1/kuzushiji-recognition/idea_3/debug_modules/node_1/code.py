import os
import sys
import shutil
import warnings
import random
import numpy as np
import pandas as pd
import torch
import cv2

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import SwinCenterNet
from library.loss import CenterNetLoss
from library.utils import kuzushiji_f1_score, decode_centernet_predictions
from library.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_mini_metadata():
    """
    Creates a small subset of the training and validation metadata
    to allow for a quick demonstration of the training loop.
    """
    print("Creating mini-datasets for demonstration...")

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Sample 12 images for training and 4 for validation
    # This ensures we have enough for a few batches (Batch Size = 2)
    mini_train = train_df.head(12).copy()
    mini_val = val_df.head(4).copy()

    # Save to working directory
    mini_train_path = os.path.join(Config.WORK_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORK_DIR, "mini_val.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)

    return mini_train_path, mini_val_path


def demo_dataset_logic():
    print("\n=== Demonstrating Dataset & Preprocessing ===")

    # Initialize dataset (uses the modified Config paths)
    dataset = KuzushijiDataset(mode="train", load_cached_data=False)

    # Fetch a single sample
    sample = dataset[0]

    # Verify Keys
    expected_keys = {"image", "hm", "ind", "wh", "reg", "cls_ids", "reg_mask"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset output: {sample.keys()}"

    # Verify Shapes
    # Image: (3, 1024, 1024)
    img_shape = sample["image"].shape
    assert img_shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_shape}"

    # Heatmap: (1, 256, 256) -> Stride 4
    hm_shape = sample["hm"].shape
    expected_hm_size = Config.IMG_SIZE // 4
    assert hm_shape == (
        1,
        expected_hm_size,
        expected_hm_size,
    ), f"Incorrect heatmap shape: {hm_shape}"

    # Verify Gaussian Heatmap Integrity
    # If there are objects, the max value in heatmap should be close to 1.0 (the peak of the gaussian)
    if sample["reg_mask"].sum() > 0:
        max_val = sample["hm"].max().item()
        assert np.isclose(
            max_val, 1.0, atol=1e-4
        ), f"Heatmap peak should be 1.0, got {max_val}"
        print(
            f"Dataset verification passed. Found {int(sample['reg_mask'].sum())} objects in sample."
        )
    else:
        print("Dataset verification passed (Sample had 0 objects).")


def demo_model_and_loss():
    print("\n=== Demonstrating Model Forward Pass & Loss ===")

    device = Config.DEVICE

    # Instantiate Model
    model = SwinCenterNet().to(device)
    model.eval()  # Set to eval for deterministic check, though we compute loss

    # Create a dummy batch
    # We use the dataset to get a real formatted batch
    dataset = KuzushijiDataset(mode="train", load_cached_data=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    images = batch["image"].to(device)

    # Forward Pass
    with torch.cuda.amp.autocast():
        outputs = model(images)

    # Verify Output Shapes
    # hm: (B, 1, H/4, W/4)
    # wh: (B, 2, H/4, W/4)
    # reg: (B, 2, H/4, W/4)
    # cls_logits: (B, NumClasses, H/4, W/4)
    B = images.size(0)
    H, W = images.shape[2], images.shape[3]
    feat_H, feat_W = H // 4, W // 4

    assert outputs["hm"].shape == (
        B,
        1,
        feat_H,
        feat_W,
    ), "Heatmap output shape mismatch"
    assert outputs["wh"].shape == (B, 2, feat_H, feat_W), "WH output shape mismatch"
    assert outputs["cls_logits"].shape == (
        B,
        Config.get_num_classes(),
        feat_H,
        feat_W,
    ), "Class logits shape mismatch"

    print("Model forward pass successful. Output shapes verified.")

    # Loss Calculation
    criterion = CenterNetLoss().to(device)

    # Move batch targets to device (handled inside loss or manually before)
    # The library loss function expects the batch dict values to be accessible,
    # but it calls .to(device) internally on them.

    total_loss, loss_stats = criterion(outputs, batch)

    print(f"Loss computed successfully: {total_loss.item():.4f}")
    print(f"Loss Components: {loss_stats}")

    assert not torch.isnan(total_loss), "Loss is NaN!"
    assert total_loss > 0, "Loss should be positive"


def demo_metric():
    print("\n=== Demonstrating Metric Logic ===")

    # Case: True Positive
    # GT: Label 'A' at box [100, 100, 50, 50] -> Center roughly (125, 125)
    # Pred: Label 'A' at point (125, 125)
    gt_str = "U+306B 100 100 50 50"
    pred_str = "U+306B 125 125"

    metrics = kuzushiji_f1_score([pred_str], [gt_str])

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert np.isclose(metrics["f1"], 1.0)
    print("Metric check (Perfect Match): Passed")

    # Case: False Positive (Wrong Label)
    pred_str_wrong = "U+9999 125 125"
    metrics = kuzushiji_f1_score([pred_str_wrong], [gt_str])
    assert metrics["tp"] == 0
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1  # The GT was missed
    print("Metric check (Wrong Label): Passed")

    # Case: False Positive (Outside Box)
    pred_str_out = "U+306B 200 200"  # Box ends at 150
    metrics = kuzushiji_f1_score([pred_str_out], [gt_str])
    assert metrics["tp"] == 0
    print("Metric check (Outside Box): Passed")


def run_training_demo():
    print("\n=== Running Trainer (Fit & Predict) ===")

    # Initialize Trainer
    # It will use the modified Config we set up globally
    trainer = Trainer()

    # Run training
    # This will train for 1 epoch on the mini dataset
    trainer.fit()

    # Run inference
    # This will generate submission.csv
    trainer.predict()

    # Verify submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated successfully with {len(sub_df)} rows.")
        print(sub_df.head())
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)

    # 2. Modify Config for Speed
    # We modify the class attributes directly to affect all modules using Config
    mini_train_path, mini_val_path = create_mini_metadata()

    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.WORK_DIR = "./working/demo_run"  # Separate work dir
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # 3. Run Demonstrations
    try:
        demo_dataset_logic()
        demo_model_and_loss()
        demo_metric()
        run_training_demo()
        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
