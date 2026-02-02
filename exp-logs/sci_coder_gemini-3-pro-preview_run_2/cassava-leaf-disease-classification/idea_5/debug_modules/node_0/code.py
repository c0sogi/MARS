import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    accuracy,
    SoftTargetCrossEntropy,
)
from library.data import (
    CassavaDataset,
    get_dataloaders,
    get_test_dataloader,
    prepare_folds,
    get_transforms,
)
from library.model import get_model
from library.engine import train_fold
from library.inference import ensemble_inference

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def create_dummy_test_metadata(original_path, new_path, num_samples=20):
    """Creates a smaller test metadata file for rapid inference testing."""
    df = pd.read_csv(original_path)
    df_small = df.head(num_samples).copy()
    df_small.to_csv(new_path, index=False)
    print(f"Created dummy test metadata at {new_path} with {len(df_small)} samples.")


def test_utils():
    print("\n--- Testing Library Utils ---")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)
    # Sum = 10*2 + 20*1 = 40. Count = 3. Avg = 13.333
    assert (
        meter.count == 3
    ), f"AverageMeter count incorrect. Expected 3, got {meter.count}"
    assert (
        abs(meter.avg - 13.3333) < 1e-4
    ), f"AverageMeter avg incorrect. Got {meter.avg}"
    print("AverageMeter: OK")

    # Test Accuracy
    # Batch size 4, 5 classes
    outputs = torch.tensor(
        [
            [0.1, 0.2, 0.5, 0.1, 0.1],  # Class 2
            [0.9, 0.0, 0.0, 0.0, 0.1],  # Class 0
            [0.1, 0.1, 0.1, 0.6, 0.1],  # Class 3
            [0.2, 0.2, 0.2, 0.2, 0.2],  # Tie/Low confidence (Class 0 by index if tie)
        ]
    )
    targets = torch.tensor([2, 0, 3, 4])  # Last one is wrong (pred 0 vs target 4)

    acc1 = accuracy(outputs, targets, topk=(1,))
    # Correct: 0, 1, 2. Incorrect: 3. Accuracy = 3/4 = 75%
    expected_acc = 75.0
    assert (
        abs(acc1[0].item() - expected_acc) < 1e-4
    ), f"Accuracy calculation incorrect. Expected {expected_acc}, got {acc1[0].item()}"
    print("Accuracy Function: OK")


def test_data(cfg):
    print("\n--- Testing Library Data ---")

    # Test Transforms
    t_train = get_transforms("train", cfg)
    t_valid = get_transforms("valid", cfg)
    assert t_train is not None and t_valid is not None
    print("Transforms: OK")

    # Test DataLoaders (using Fold 0)
    # This triggers prepare_folds internally
    train_loader, val_loader = get_dataloaders(0, cfg)

    # Check batch size
    images, labels = next(iter(train_loader))
    assert (
        images.shape[0] == cfg.batch_size
    ), f"Train batch size mismatch. Expected {cfg.batch_size}, got {images.shape[0]}"
    assert images.shape[1] == 3, "Image channels mismatch. Expected 3."
    assert images.shape[2] == cfg.image_size, "Image height mismatch."
    assert images.shape[3] == cfg.image_size, "Image width mismatch."
    assert labels.shape[0] == cfg.batch_size, "Label batch size mismatch."

    print(f"DataLoader: OK (Batch shape: {images.shape})")


def test_model(cfg):
    print("\n--- Testing Library Model ---")

    # Use a lightweight model for the test to save memory/time
    # Note: We rely on timm to create it.
    model = get_model(cfg, pretrained=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, cfg.image_size, cfg.image_size)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        cfg.num_classes,
    ), f"Model output shape incorrect. Expected (2, {cfg.num_classes}), got {output.shape}"
    print("Model Forward Pass: OK")


def test_training_and_inference(cfg):
    print("\n--- Testing Training and Inference Workflow ---")

    # Train Fold 0 and Fold 1
    # We set n_folds=2 in config, so we loop range(2)
    for fold in range(cfg.n_folds):
        print(f"Testing training for Fold {fold}...")
        train_fold(fold, cfg)

        # Verify checkpoint exists
        ckpt_path = os.path.join(cfg.working_dir, f"fold_{fold}_best.pth")
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint not found for fold {fold} at {ckpt_path}"
        print(f"Checkpoint verified: {ckpt_path}")

    # Inference
    print("Testing Ensemble Inference...")
    ensemble_inference(cfg)

    # Verify submission
    sub_path = os.path.join(cfg.submission_dir, "submission.csv")
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Verify rows match dummy test set
    df_test_meta = pd.read_csv(cfg.test_metadata_path)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count mismatch. Expected {len(df_test_meta)}, got {len(df_sub)}"
    assert (
        "image_id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing."

    print("Workflow: OK")


def main():
    # 1. Setup Configuration
    # We modify the Config class attributes directly where necessary to ensure
    # static methods in library.data use the correct paths.
    demo_dir = "./working/demo_execution_script"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Set global config paths
    Config.working_dir = demo_dir
    Config.submission_dir = demo_dir

    # Create a dummy test metadata file to make inference fast
    dummy_test_path = os.path.join(demo_dir, "test_dummy.csv")
    create_dummy_test_metadata(
        Config.test_metadata_path, dummy_test_path, num_samples=16
    )
    Config.test_metadata_path = dummy_test_path

    # Instantiate and override instance config for speed
    cfg = Config()
    cfg.debug = True
    cfg.debug_sample_size = 32  # Small sample size for fast epochs
    cfg.epochs = 1  # 1 Epoch
    cfg.n_folds = 2  # 2 Folds to test ensemble logic
    cfg.batch_size = 8  # Small batch size
    cfg.num_workers = 2  # Reduce workers for simple script
    cfg.model_name = "resnet18"  # Use a lighter model for demonstration speed

    # Seed
    seed_everything(cfg.seed)

    print(
        f"Configuration: Debug={cfg.debug}, Epochs={cfg.epochs}, Folds={cfg.n_folds}, Device={cfg.device}"
    )

    # 2. Run Tests
    test_utils()
    test_data(cfg)
    test_model(cfg)
    test_training_and_inference(cfg)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
