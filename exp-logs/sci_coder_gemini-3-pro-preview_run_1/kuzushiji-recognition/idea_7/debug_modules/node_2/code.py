import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import LabelEncoder, decode_predictions
from library.dataset import KuzushijiDataset
from library.model import ConvNeXtCenterNet
from library.loss import CenterNetLoss
from library.train import Trainer


def run_demo():
    print("=== Starting Kuzushiji Pipeline Demo ===")

    # 1. Setup & Configuration Overrides for Speed
    # We modify the Config class directly to run a minimal version of the task
    print("Configuring for fast execution...")
    seed_everything(42)

    # Create a specific working directory for this demo
    demo_work_dir = "./working/demo_run"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir, exist_ok=True)

    Config.WORK_DIR = demo_work_dir
    Config.BEST_MODEL_PATH = os.path.join(demo_work_dir, "best_model.pth")
    Config.LABEL_ENCODER_PATH = os.path.join(demo_work_dir, "label_encoder.npy")

    # Enable Debug Mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small enough for CPU/Fast GPU run
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 2. Label Encoder Verification
    print("\n--- Testing Label Encoder ---")
    le = LabelEncoder()
    # Force fit from metadata to ensure logic works (ignoring existing cache for demo)
    le.fit(load_cached_data=False)

    # Test transformation
    test_char = le.classes_[0]  # Get first available class
    encoded = le.transform(test_char)
    decoded = le.inverse_transform(encoded)

    assert encoded != -1, "LabelEncoder failed to transform a valid character."
    assert (
        decoded == test_char
    ), f"LabelEncoder inverse transform mismatch: {decoded} != {test_char}"
    print(f"LabelEncoder verified. Total classes: {len(le)}")

    # 3. Dataset Verification
    print("\n--- Testing Dataset ---")
    # Initialize dataset (uses Config.DEBUG settings)
    dataset = KuzushijiDataset(mode="train", load_cached_data=False)

    # Fetch a single sample
    sample = dataset[0]

    # Verify keys
    required_keys = [
        "image",
        "hm",
        "ind",
        "wh",
        "reg",
        "cls_id",
        "reg_mask",
        "image_id",
    ]
    for key in required_keys:
        assert key in sample, f"Dataset sample missing key: {key}"

    # Verify Shapes
    img_tensor = sample["image"]
    hm_tensor = sample["hm"]

    # Image should be (3, 1024, 1024) based on Config.IMG_SIZE
    assert img_tensor.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_tensor.shape}"

    # Heatmap should be downsampled by 4 (1, 256, 256)
    expected_out_size = Config.IMG_SIZE // 4
    assert hm_tensor.shape == (
        1,
        expected_out_size,
        expected_out_size,
    ), f"Incorrect heatmap shape: {hm_tensor.shape}"

    print("Dataset verified. Sample shapes correct.")

    # 4. Model Architecture Verification
    print("\n--- Testing Model Architecture ---")
    model = ConvNeXtCenterNet(
        num_classes=Config.NUM_CLASSES, pretrained=False
    )  # No pretrained weights download for speed if possible, or cached
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(dummy_input)

    # Check output keys
    assert "hm" in outputs and "reg_wh" in outputs and "cls_logits" in outputs

    # Check output shapes
    # hm: (B, 1, H/4, W/4)
    assert outputs["hm"].shape == (2, 1, expected_out_size, expected_out_size)
    # reg_wh: (B, 4, H/4, W/4)
    assert outputs["reg_wh"].shape == (2, 4, expected_out_size, expected_out_size)
    # cls_logits: (B, NumClasses, H/4, W/4)
    assert outputs["cls_logits"].shape == (
        2,
        Config.NUM_CLASSES,
        expected_out_size,
        expected_out_size,
    )

    print("Model forward pass successful. Output shapes correct.")

    # 5. Loss Function Verification
    print("\n--- Testing Loss Function ---")
    criterion = CenterNetLoss()

    # Construct a batch dictionary from the dataset sample
    # We need to add the batch dimension
    batch_sample = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch_sample[k] = v.unsqueeze(0).to(Config.DEVICE)  # Add batch dim

    # Get model outputs for this sample
    with torch.no_grad():
        sample_out = model(batch_sample["image"])

    # Calculate loss
    loss, loss_stats = criterion(sample_out, batch_sample)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"Loss calculation successful. Total Loss: {loss.item():.4f}")
    print(f"Loss Stats: {loss_stats}")

    # 6. Decoding Verification
    print("\n--- Testing Prediction Decoding ---")
    hm = torch.sigmoid(sample_out["hm"])
    reg = sample_out["reg_wh"][:, 0:2, :, :]
    wh = sample_out["reg_wh"][:, 2:4, :, :]

    dets = decode_predictions(hm, reg, wh, K=10)
    # Shape: (Batch, K, 6) -> (1, 10, 6)
    assert dets.shape == (1, 10, 6), f"Decoding shape mismatch: {dets.shape}"
    print("Decoding successful.")

    # 7. Full Training Loop Verification
    print("\n--- Testing Trainer (1 Epoch, Debug Subset) ---")
    # Initialize Trainer
    # Note: Trainer initializes datasets internally. Since we set Config.DEBUG=True,
    # it will use the subset.
    trainer = Trainer()

    # Run training
    trainer.fit()

    # Verify artifacts
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    log_path = os.path.join(Config.WORK_DIR, "training_log.csv")
    assert os.path.exists(log_path), "Training log was not created."

    df_log = pd.read_csv(log_path)
    assert len(df_log) == 1, "Log should contain exactly 1 epoch."
    print("Training loop completed successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
