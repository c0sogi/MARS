import os
import sys
import torch
import numpy as np
import pandas as pd
import random
import warnings
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import SwinCenterNet
from library.loss import CenterNetLoss
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import decode_center_net


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Setup & Configuration Overrides for Speed
    set_seed(Config.SEED)

    # Override Config for fast demonstration
    print("Configuring for debug mode...")
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 2  # Reduced from 4 to avoid OOM with Swin-Base @ 1024x1024
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Ensure working directories exist
    Config.setup()

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Dataset Verification
    print("\n--- Verifying Dataset ---")
    dataset = KuzushijiDataset(split="train", debug_size=Config.DEBUG_SAMPLE_SIZE)
    print(f"Dataset size: {len(dataset)}")

    # Fetch one sample
    sample = dataset[0]

    # Check keys
    expected_keys = {
        "image",
        "heatmap",
        "wh",
        "reg",
        "ind",
        "cls_ids",
        "mask",
        "image_id",
    }
    assert (
        set(sample.keys()) == expected_keys
    ), f"Missing keys in dataset sample. Found: {sample.keys()}"

    # Check Shapes
    # Image: (3, 1024, 1024)
    assert sample["image"].shape == (
        3,
        1024,
        1024,
    ), f"Incorrect image shape: {sample['image'].shape}"

    # Heatmap: (1, 256, 256) -> Output stride is 4
    out_dim = 1024 // Config.OUTPUT_STRIDE
    assert sample["heatmap"].shape == (
        1,
        out_dim,
        out_dim,
    ), f"Incorrect heatmap shape: {sample['heatmap'].shape}"

    # Dense Regression maps: (2, 256, 256)
    assert sample["wh"].shape == (
        2,
        out_dim,
        out_dim,
    ), f"Incorrect wh shape: {sample['wh'].shape}"

    # Sparse Targets
    # ind, cls_ids, mask should have length equal to max_objs (set in dataset, default 1200 usually, checked in dataset.py)
    # In dataset.py: self.max_objs = 1200
    max_objs = 1200
    assert sample["ind"].shape == (
        max_objs,
    ), f"Incorrect ind shape: {sample['ind'].shape}"

    print("Dataset verification passed.")

    # 3. Model & Loss Verification
    print("\n--- Verifying Model and Loss ---")
    model = SwinCenterNet().to(device)
    criterion = CenterNetLoss()

    # Prepare a batch
    # We need to stack the sample to make a batch of size 2 for testing
    collate_fn = torch.utils.data.dataloader.default_collate
    batch_list = [dataset[0], dataset[1]]
    batch = collate_fn(batch_list)

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Forward Pass
    outputs = model(batch["image"])

    # Check Output Keys
    assert "hm" in outputs
    assert "wh" in outputs
    assert "reg" in outputs
    assert "cls" in outputs

    # Check Output Shapes
    # hm: (B, 1, H/4, W/4)
    assert outputs["hm"].shape == (
        2,
        1,
        out_dim,
        out_dim,
    ), f"Model HM shape mismatch: {outputs['hm'].shape}"
    # cls: (B, NumClasses, H/4, W/4)
    assert outputs["cls"].shape == (
        2,
        Config.NUM_CLASSES,
        out_dim,
        out_dim,
    ), f"Model CLS shape mismatch: {outputs['cls'].shape}"

    print("Model forward pass successful.")

    # Loss Calculation
    loss, loss_stats = criterion(outputs, batch)

    # Check Loss validity
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"Loss calculation successful. Initial Loss: {loss.item():.4f}")
    print(
        f"Breakdown: HM={loss_stats['hm_loss']:.4f}, WH={loss_stats['wh_loss']:.4f}, CLS={loss_stats['cls_loss']:.4f}"
    )

    # --- FIX: Cleanup Verification Resources to prevent OOM ---
    print("Cleaning up verification resources...")
    del model, outputs, batch, loss, loss_stats
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------

    # 4. Trainer Demonstration (Training Loop)
    print("\n--- Running Trainer (1 Epoch) ---")
    trainer = Trainer(debug=True)

    # We already overrode Config.NUM_EPOCHS = 1 and DEBUG_SAMPLE_SIZE = 20
    trainer.fit()

    # Verify artifacts
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    assert os.path.exists(Config.LOG_PATH), "Training log was not saved."

    # Check log content
    log_df = pd.read_csv(Config.LOG_PATH)
    assert len(log_df) == 1, "Log should contain exactly 1 epoch entry."
    print("Training loop completed successfully.")

    # 5. Inference Demonstration
    print("\n--- Running Inference ---")
    # Generate submission using the model we just trained (saved at Config.BEST_MODEL_PATH)
    generate_submission(weights_path=Config.BEST_MODEL_PATH, debug_size=10)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check columns
    assert (
        "image_id" in sub_df.columns and "labels" in sub_df.columns
    ), "Submission columns missing."

    # Check format of a label if not empty
    # Note: With 1 epoch on 20 samples, the model might not predict anything (empty string),
    # or predict garbage. We just check the file structure is valid.
    if len(sub_df) > 0:
        first_label = sub_df.iloc[0]["labels"]
        if isinstance(first_label, str) and len(first_label) > 0:
            parts = first_label.split(" ")
            # Should be divisible by 3 (Char X Y)
            assert len(parts) % 3 == 0, "Label format should be 'Char X Y' repeated."

    print("Inference completed successfully.")
    print("\n--- All Demonstrations Passed ---")


if __name__ == "__main__":
    main()
