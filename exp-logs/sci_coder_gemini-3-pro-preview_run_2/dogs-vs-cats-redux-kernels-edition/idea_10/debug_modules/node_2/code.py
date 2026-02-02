import os
import sys
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.augmentations import get_train_transforms, get_valid_transforms
from library.dataset import get_dataset, DogCatDataset
from library.models import create_model
from library.engine import fit
from library.stacking import StackingTrainer, predict_stacking


def run_demo():
    # =========================================================================
    # 1. Setup & Configuration Overrides
    # =========================================================================
    print(">>> 1. Setting up configuration and environment...")
    seed_everything(Config.SEED)

    # Define demo working directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.OOF_DIR = os.path.join(DEMO_DIR, "oof")
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup_directories()

    # =========================================================================
    # 2. Data Preparation (Subsets)
    # =========================================================================
    print(">>> 2. Preparing data subsets for speed...")

    # Load original metadata
    orig_train_df = pd.read_csv("./metadata/train.csv")
    orig_val_df = pd.read_csv("./metadata/val.csv")
    orig_test_df = pd.read_csv("./metadata/test.csv")

    # Create subsets (20 samples for train, 10 for val, 10 for test)
    subset_train = orig_train_df.head(20).copy()
    subset_val = orig_val_df.head(10).copy()
    subset_test = orig_test_df.head(10).copy()

    # Save subsets to demo directory
    demo_train_csv = os.path.join(DEMO_DIR, "train_subset.csv")
    demo_val_csv = os.path.join(DEMO_DIR, "val_subset.csv")
    demo_test_csv = os.path.join(DEMO_DIR, "test_subset.csv")

    subset_train.to_csv(demo_train_csv, index=False)
    subset_val.to_csv(demo_val_csv, index=False)
    subset_test.to_csv(demo_test_csv, index=False)

    # Override Config metadata paths to point to subsets
    Config.TRAIN_CSV = demo_train_csv
    Config.VAL_CSV = demo_val_csv
    Config.TEST_CSV = demo_test_csv

    print(
        f"    Created subsets: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # =========================================================================
    # 3. Dataset & Transforms Demo
    # =========================================================================
    print(">>> 3. Demonstrating Dataset and Transforms...")

    # Instantiate datasets
    train_ds = get_dataset("train", transforms=get_train_transforms(image_size=224))
    val_ds = get_dataset("val", transforms=get_valid_transforms(image_size=224))

    # Verify lengths
    assert len(train_ds) == 20, f"Expected 20 training samples, got {len(train_ds)}"
    assert len(val_ds) == 10, f"Expected 10 validation samples, got {len(val_ds)}"

    # Verify item retrieval
    img, label = train_ds[0]
    assert isinstance(img, torch.Tensor), "Dataset should return a Tensor image"
    assert img.shape == (
        3,
        224,
        224,
    ), f"Expected image shape (3, 224, 224), got {img.shape}"
    assert isinstance(
        label, (int, np.integer)
    ), f"Label should be integer, got {type(label)}"

    print("    Dataset verification successful.")

    # =========================================================================
    # 4. Model & Training Engine Demo
    # =========================================================================
    print(">>> 4. Demonstrating Model Training (1 Epoch)...")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Initialize Model (using one from Config.MODEL_NAMES)
    # Using pretrained=False to avoid download time/errors in this demo
    model_name = Config.MODEL_NAMES[1]  # swin_small_patch4_window7_224
    print(f"    Initializing model: {model_name}")
    model = create_model(model_name, num_classes=1, pretrained=False)
    model.to(Config.DEVICE)

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)

    # Run Training
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{model_name}.pth")

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=1,
        patience=1,
        checkpoint_path=checkpoint_path,
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"    Training complete. Checkpoint saved to {checkpoint_path}")

    # =========================================================================
    # 5. Stacking Pipeline Demo
    # =========================================================================
    print(">>> 5. Demonstrating Stacking Pipeline...")

    # 5a. Generate Dummy OOF Predictions for all models in Config
    # The StackingTrainer expects files in Config.OOF_DIR
    print("    Generating dummy OOF and Test predictions...")

    for m_name in Config.MODEL_NAMES:
        # Dummy OOF: Needs 'filepath' and prediction column
        # We use the subset_train dataframe
        dummy_oof = subset_train[["filepath"]].copy()
        # Random probabilities
        dummy_oof["pred"] = np.random.uniform(0, 1, len(dummy_oof))

        # Save as consolidated OOF file
        oof_file = os.path.join(Config.OOF_DIR, f"{m_name}_oof.csv")
        dummy_oof.to_csv(oof_file, index=False)

        # Dummy Test Preds: Needs 'id' and prediction column
        dummy_test = subset_test[["id"]].copy()
        dummy_test["pred"] = np.random.uniform(0, 1, len(dummy_test))

        # Save as consolidated Test file
        test_file = os.path.join(Config.OOF_DIR, f"{m_name}_test.csv")
        dummy_test.to_csv(test_file, index=False)

    # 5b. Train Meta Learner
    print("    Training Meta-Learner...")
    stacker = StackingTrainer()
    # load_cached_data=False forces it to read the CSVs we just made
    clf = stacker.train(load_cached_data=False)

    assert os.path.exists(stacker.meta_model_path), "Meta-model file not found!"

    # 5c. Predict Stacking (Inference)
    print("    Running Stacking Inference...")
    predict_stacking(load_cached_data=False)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found!"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    assert len(sub_df) == len(
        subset_test
    ), f"Submission length mismatch. Expected {len(subset_test)}, got {len(sub_df)}"
    assert (
        "id" in sub_df.columns and "label" in sub_df.columns
    ), "Submission missing required columns"

    print(f"    Submission generated successfully at {Config.SUBMISSION_FILE}")
    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
