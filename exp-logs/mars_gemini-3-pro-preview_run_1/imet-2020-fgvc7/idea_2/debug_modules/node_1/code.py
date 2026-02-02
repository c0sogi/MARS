import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_micro_f1, AverageMeter
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkModel
from library.trainer import Trainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("============================================================")
    print("Artwork Attribute Labeling: Library Demo & Verification")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup for Fast Demonstration
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config for speed and isolation
    Config.debug = True
    Config.debug_sample_size = 100  # Use only 100 samples
    Config.epochs = 2  # Train for only 2 epochs
    Config.batch_size = 16  # Small batch size
    Config.num_workers = 2  # Reduce worker overhead
    Config.pretrained = False  # Disable downloading weights for speed/offline safety
    Config.output_dir = "./working/demo_output"
    Config.model_save_path = os.path.join(Config.output_dir, "demo_model.pth")
    Config.submission_path = os.path.join(Config.output_dir, "submission.csv")

    # Ensure clean slate
    if os.path.exists(Config.output_dir):
        shutil.rmtree(Config.output_dir)
    Config.setup()

    seed_everything(Config.seed)
    print("Configuration updated for demo mode.")

    # ------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)
    # Total sum = 10*2 + 20*1 = 40, Total count = 3, Avg = 13.333
    assert meter.count == 3, f"AverageMeter count mismatch: {meter.count}"
    assert abs(meter.avg - 13.3333) < 1e-3, f"AverageMeter avg mismatch: {meter.avg}"
    print("-> AverageMeter passed.")

    # Test calculate_micro_f1
    # Create dummy logits (N=2, C=3) and targets
    # Logits: Sample 0 -> [High, Low, High], Sample 1 -> [Low, High, Low]
    logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
    # Targets: Matches Sample 0, Mismatch Sample 1
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    # Preds will be [[1, 0, 1], [0, 1, 0]]
    # Targets are   [[1, 0, 1], [0, 0, 1]]
    # TP: (0,0), (0,2) = 2
    # FP: (1,1) = 1
    # FN: (1,2) = 1
    # Precision = 2 / (2+1) = 0.666
    # Recall = 2 / (2+1) = 0.666
    # F1 = 0.666

    f1 = calculate_micro_f1(logits, targets, threshold=0.5)
    assert abs(f1 - 2 / 3) < 1e-3, f"F1 Score calculation mismatch: {f1}"
    print("-> calculate_micro_f1 passed.")

    # ------------------------------------------------------------------
    # 3. Verify Dataset Loading and Processing
    # ------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")

    # Initialize Train Dataset
    train_ds = ArtworkDataset(
        csv_path=Config.train_metadata_path,
        mode="train",
        transform=get_transforms(data_split="train"),
        load_cached_data=False,  # Force reload to test logic
    )

    print(f"-> Train dataset size (debug): {len(train_ds)}")
    assert (
        len(train_ds) == Config.debug_sample_size
    ), "Dataset size does not match debug limit."

    # Fetch one sample
    img, target = train_ds[0]

    # Check Image Shape: (C, H, W) -> (3, 320, 320)
    assert img.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), f"Image shape mismatch: {img.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a tensor."

    # Check Target Shape: (NumClasses,)
    assert target.shape == (
        Config.num_classes,
    ), f"Target shape mismatch: {target.shape}"

    # Check Test Dataset
    test_ds = ArtworkDataset(
        csv_path=Config.test_metadata_path,
        mode="test",
        transform=get_transforms(data_split="test"),
        load_cached_data=False,
    )
    test_img, test_id = test_ds[0]
    assert isinstance(test_id, str), "Test ID should be a string."
    print("-> Dataset structure verified.")

    # ------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = ArtworkModel(pretrained=False)
    model.to(Config.device)
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(
        Config.device
    )

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch, NumClasses)
    assert output.shape == (
        2,
        Config.num_classes,
    ), f"Model output shape mismatch: {output.shape}"

    print(f"-> Model forward pass successful. Output shape: {output.shape}")

    # ------------------------------------------------------------------
    # 5. Execute Training and Inference (Trainer)
    # ------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Trainer)...")

    trainer = Trainer()

    # Run Training
    # This calls fit(), which runs train_one_epoch and validate for Config.epochs
    trainer.fit()

    # Check if model checkpoint was saved
    if os.path.exists(Config.model_save_path):
        print(f"-> Model checkpoint successfully saved at {Config.model_save_path}")
    else:
        # If model didn't improve (unlikely with random init vs 0 score), it might not save.
        # But with F1 starting at 0, any valid prediction improves it.
        # However, if validation fails to find any positive samples, score remains 0.
        # In debug mode with random weights, this is possible.
        print("-> Note: Best model might not be saved if F1 score was 0.0 throughout.")

    print("\n[6] Executing Inference...")

    # Run Inference
    trainer.predict()

    # Verify Submission
    if os.path.exists(Config.submission_path):
        sub_df = pd.read_csv(Config.submission_path)
        print(f"-> Submission file generated at {Config.submission_path}")
        print(f"-> Submission rows: {len(sub_df)}")
        print("-> Sample submission head:")
        print(sub_df.head(3))

        expected_cols = ["id", "attribute_ids"]
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch: {sub_df.columns}"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n============================================================")
    print("Demo execution completed successfully.")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
