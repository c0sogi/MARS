import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_iou
from library.layers import BlurPool
from library.dataset import ChestXrayDataset, get_transforms
from library.model import AntiAliasedResNetUNet
from library.trainer import Trainer


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Configuration Override for Demo
    # We modify the Config class directly to optimize for speed and set a clean working environment.
    print("[1/5] Configuring environment...")
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.IMAGE_SIZE = (224, 224)  # Reduced size for faster processing
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Clean up demo directory if it exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration set. Debug mode enabled.")

    # 2. Utility Verification
    print("\n[2/5] Verifying Utilities...")
    # Test IoU Calculation
    # Box format: xmin, ymin, xmax, ymax
    # Box 1: 10x10 square at (0,0) -> Area 100
    # Box 2: 10x10 square at (5,0) -> Area 100
    # Intersection: 5x10 rectangle -> Area 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 0.333...
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
    box2 = torch.tensor([[5, 0, 15, 10]], dtype=torch.float)
    iou = calculate_iou(box1, box2)

    print(f"Calculated IoU: {iou.item():.4f}")
    assert torch.abs(iou - 0.3333) < 0.001, "IoU calculation failed validation."
    print("Utils verification passed.")

    # 3. Layer Verification
    print("\n[3/5] Verifying Layers (BlurPool)...")
    blur_layer = BlurPool(channels=3, stride=2)
    # Input: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = blur_layer(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # BlurPool with stride 2 should halve the spatial dimensions
    assert output.shape == (
        1,
        3,
        16,
        16,
    ), f"BlurPool output shape mismatch. Expected (1, 3, 16, 16), got {output.shape}"
    print("Layer verification passed.")

    # 4. Dataset and Model Verification
    print("\n[4/5] Verifying Dataset and Model...")

    # Load a tiny subset of metadata manually
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH).head(4)

    # Instantiate Dataset
    # Note: This will trigger _process_and_cache, creating .npy files in Config.WORKING_DIR
    ds = ChestXrayDataset(
        metadata_df=df_train,
        mode="train",
        transform=get_transforms("train"),
        cache_dir=Config.WORKING_DIR,
        load_cached_data=False,
    )

    assert len(ds) == 4, "Dataset length mismatch."

    # Fetch one sample
    sample = ds[0]
    image = sample["image"]
    mask = sample["mask"]
    study_label = sample["study_label"]

    # Verify Data Shapes
    # Image: (3, H, W) due to ToTensorV2 (assuming RGB conversion in dataset)
    # Mask: (1, H, W)
    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    assert image.shape == (3, 224, 224), f"Image tensor shape incorrect: {image.shape}"
    assert mask.shape == (1, 224, 224), f"Mask tensor shape incorrect: {mask.shape}"
    assert study_label.shape == (
        4,
    ), f"Study label shape incorrect: {study_label.shape}"

    # Instantiate Model
    model = AntiAliasedResNetUNet()

    # Create a batch (add batch dimension)
    batch_image = image.unsqueeze(0).float()  # (1, 3, 224, 224)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        cls_logits, seg_logits = model(batch_image)

    print(f"Model Cls Output: {cls_logits.shape}")
    print(f"Model Seg Output: {seg_logits.shape}")

    # Verify Model Output Shapes
    # Cls: (Batch, Num_Classes) -> (1, 4)
    # Seg: (Batch, Num_Classes_Img, H, W) -> (1, 1, 224, 224)
    assert cls_logits.shape == (1, 4), "Classification output shape mismatch."
    assert seg_logits.shape == (1, 1, 224, 224), "Segmentation output shape mismatch."

    print("Dataset and Model verification passed.")

    # 5. Trainer Integration Test
    print("\n[5/5] Running Trainer Integration Test (1 Epoch)...")

    # Instantiate Trainer
    # This will initialize datasets internally using Config.TRAIN_METADATA_PATH
    # and Config.DEBUG=True logic.
    trainer = Trainer()

    # Run training
    best_score = trainer.fit()

    # Verify artifacts
    print(f"Training finished. Best score: {best_score}")
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
