import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_folds, AppleDataset, get_transforms
from library.modeling import get_model
from library.engine import fit_model
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Apple Disease Detection Pipeline Demo ====")

    # 1. Setup and Configuration Override
    # We modify the Config class at runtime to ensure the demo runs quickly
    # within the resource and time constraints.
    print("[1] Configuring environment...")

    Config.SEED = 42
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_FOLDS = 2  # K-Fold requires at least 2 splits. We still only run Fold 0.
    Config.BATCH_SIZE = 4  # Small batch size for debug
    Config.DEBUG_SAMPLE_SIZE = 16  # Small dataset size
    Config.PATIENCE = 1  # Early stopping

    # Set a specific working directory for this demo
    Config.WORK_DIR = "./working/demo_test"
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Update submission path to be inside the working directory for safety
    Config.SUBMISSION_PATH = os.path.join(
        Config.WORK_DIR, "submission", "submission.csv"
    )

    print(f"    Work Dir: {Config.WORK_DIR}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Epochs: {Config.EPOCHS}")

    # 2. Data Preparation
    print("\n[2] Preparing Data...")

    # Get debug folds (returns a small sampled dataframe)
    df = get_folds(debug=True)
    print(f"    Loaded debug dataframe with shape: {df.shape}")

    # We simulate Fold 0 split
    fold = 0
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Handle edge case where debug split might result in empty train/val if size is too small
    # For this demo, we force a split if needed
    if len(train_df) == 0 or len(val_df) == 0:
        print("    Adjusting split for very small debug set...")
        train_df = df.iloc[:12].copy()
        val_df = df.iloc[12:].copy()

    print(f"    Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Create Datasets
    # We use 'effnet' transforms initially just to verify data loading
    train_dataset = AppleDataset(
        train_df, transform=get_transforms(data="train", model_type="effnet")
    )
    val_dataset = AppleDataset(
        val_df, transform=get_transforms(data="valid", model_type="effnet")
    )

    # Create Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debug execution
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Verify Data Loading
    images, labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    assert images.shape[0] == Config.BATCH_SIZE or images.shape[0] == len(
        train_df
    ), "Batch size mismatch"
    assert images.shape[1] == 3, "Image should have 3 channels"

    # 3. Model Training
    # We will train both architectures defined in Config to ensure generate_submission works
    architectures = ["effnet", "maxvit"]
    device = torch.device(Config.DEVICE)

    for arch in architectures:
        print(f"\n[3] Training Model: {arch}...")

        # Get appropriate transforms for this architecture
        # (Image sizes differ: effnet=380, maxvit=224)
        t_train_ds = AppleDataset(train_df, transform=get_transforms("train", arch))
        t_val_ds = AppleDataset(val_df, transform=get_transforms("valid", arch))

        t_train_loader = torch.utils.data.DataLoader(
            t_train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        t_val_loader = torch.utils.data.DataLoader(
            t_val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Initialize Model
        model = get_model(model_type=arch, pretrained=True)
        model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train (fit_model saves the best checkpoint to Config.WORK_DIR)
        best_score = fit_model(
            model=model,
            train_loader=t_train_loader,
            val_loader=t_val_loader,
            optimizer=optimizer,
            device=device,
            fold=fold,
            model_name=arch,
        )

        print(f"    {arch} training complete. Best AUC: {best_score:.4f}")

        # Verify checkpoint exists
        ckpt_path = os.path.join(Config.WORK_DIR, f"{arch}_fold_{fold}_best.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # 4. Inference and Submission
    print("\n[4] Generating Submission...")

    # generate_submission uses Config.TEST_CSV.
    # It will look for models in Config.WORK_DIR.
    # We use debug=True so it subsamples the test set.
    generate_submission(debug=True)

    # 5. Verification
    print("\n[5] Verifying Output...")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file found at {Config.SUBMISSION_PATH}")
        print(f"    Shape: {sub_df.shape}")
        print(f"    Columns: {list(sub_df.columns)}")

        # Assertions
        expected_cols = ["image_id"] + Config.CLASS_LABELS
        assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
        assert (
            len(sub_df) == Config.DEBUG_SAMPLE_SIZE
        ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in debug mode"

        # Check values are probabilities
        prob_cols = Config.CLASS_LABELS
        assert (sub_df[prob_cols].values >= 0).all() and (
            sub_df[prob_cols].values <= 1
        ).all(), "Probabilities out of range"

        print("    Verification Successful!")
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )


if __name__ == "__main__":
    run_demo()
