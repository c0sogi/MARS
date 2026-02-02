import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path for module imports
sys.path.append(".")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import CassavaDataset, get_transforms, mixup_cutmix_fn
from library.model import CassavaClassifier
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import generate_submission


def run_demo():
    print("=== Starting Cassava Leaf Disease Classification Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration and Environment...")

    # Set deterministic seed
    seed_everything(42)

    # Override Config for a fast demo run
    # We use a lightweight model (ResNet18) instead of ViT/BEiT to save time/memory
    Config.working_dir = "./working/demo_run"
    Config.model_a_name = "resnet18"
    Config.model_b_name = "resnet18"
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.num_workers = 2  # Reduce workers for small data

    # Setup directories
    Config.setup()

    print(f"Working Directory: {Config.working_dir}")
    print(f"Device: {Config.device}")

    # ---------------------------------------------------------
    # 2. Prepare Demo Data (Subsets)
    # ---------------------------------------------------------
    print("\n[2] Preparing Demo Metadata (Subsets)...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create tiny subsets (12 samples for train to allow batching, 4 for val/test)
    demo_train_df = orig_train.head(12).copy()
    demo_val_df = orig_val.head(4).copy()
    demo_test_df = orig_test.head(4).copy()

    # Save demo metadata
    demo_train_path = os.path.join(Config.working_dir, "train_meta.parquet")
    demo_val_path = os.path.join(Config.working_dir, "val_meta.parquet")
    demo_test_path = os.path.join(Config.working_dir, "test_meta.parquet")

    # Saving as CSV for compatibility with CassavaDataset which reads CSV
    # Note: Variable names say parquet but we use csv as per library requirement
    demo_train_path = demo_train_path.replace(".parquet", ".csv")
    demo_val_path = demo_val_path.replace(".parquet", ".csv")
    demo_test_path = demo_test_path.replace(".parquet", ".csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    # Update Config paths to point to demo metadata
    Config.train_metadata = demo_train_path
    Config.val_metadata = demo_val_path
    Config.test_metadata = demo_test_path
    Config.submission_path = os.path.join(Config.working_dir, "submission_demo.csv")

    print(f"Created demo train set: {len(demo_train_df)} rows")
    print(f"Created demo val set:   {len(demo_val_df)} rows")
    print(f"Created demo test set:  {len(demo_test_df)} rows")

    # ---------------------------------------------------------
    # 3. Verify Data Loading & Augmentation
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loading and Augmentation...")

    # Instantiate Dataset
    train_dataset = CassavaDataset(
        metadata_path=Config.train_metadata,
        transform=get_transforms("train", img_size=Config.img_size),
        is_train=True,
    )

    # Assertions
    assert len(train_dataset) == 12, "Dataset length mismatch"

    # Fetch one sample
    img, label = train_dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")

    assert img.shape == (
        3,
        Config.img_size,
        Config.img_size,
    ), "Incorrect image tensor shape"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # Verify MixUp/CutMix Logic
    print("Verifying MixUp/CutMix function...")
    dummy_imgs = torch.randn(4, 3, 224, 224)
    dummy_lbls = torch.tensor([0, 1, 2, 3])
    mixed_imgs, t_a, t_b, lam = mixup_cutmix_fn(
        dummy_imgs, dummy_lbls, alpha=1.0, prob=0.5
    )

    assert mixed_imgs.shape == dummy_imgs.shape, "Mixed image shape mismatch"
    assert t_a.shape == dummy_lbls.shape, "Target A shape mismatch"
    assert t_b.shape == dummy_lbls.shape, "Target B shape mismatch"
    print("MixUp/CutMix verification passed.")

    # ---------------------------------------------------------
    # 4. Model Initialization & Training Loop
    # ---------------------------------------------------------
    print("\n[4] Initializing Model and Running Training Loop...")

    device = torch.device(Config.device)

    # Initialize Model (using ResNet18 for demo speed)
    model = CassavaClassifier(
        model_name=Config.model_a_name, num_classes=Config.num_classes, pretrained=True
    )
    model.to(device)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_dataset = CassavaDataset(
        metadata_path=Config.val_metadata,
        transform=get_transforms("valid", img_size=Config.img_size),
        is_train=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Optimizer & Scaler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()

    # Train for 1 Epoch
    print("Training for 1 epoch...")
    avg_loss = train_one_epoch(1, model, train_loader, optimizer, scaler, device)
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Validate
    print("Validating...")
    val_loss, val_acc = valid_one_epoch(1, model, val_loader, device)
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of bounds"

    # Save the demo model
    demo_model_path = os.path.join(Config.working_dir, "resnet18_demo.pth")
    torch.save(model.state_dict(), demo_model_path)
    print(f"Demo model saved to {demo_model_path}")

    # ---------------------------------------------------------
    # 5. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n[5] Running Inference and Generating Submission...")

    # We will use the same demo model for both 'model_a' and 'model_b' slots
    # just to demonstrate the ensemble logic in generate_submission

    try:
        generate_submission(
            model_a_path=demo_model_path,
            model_b_path=demo_model_path,
            output_path=Config.submission_path,
        )
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify Submission
    if os.path.exists(Config.submission_path):
        sub_df = pd.read_csv(Config.submission_path)
        print(f"Submission generated with shape: {sub_df.shape}")

        # Assertions
        assert sub_df.shape == (4, 2), f"Expected (4, 2) but got {sub_df.shape}"
        assert list(sub_df.columns) == [
            "image_id",
            "label",
        ], "Incorrect columns in submission"
        assert (
            sub_df["label"].dtype == np.int64 or sub_df["label"].dtype == np.int32
        ), "Label column should be integer"
        print("Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
