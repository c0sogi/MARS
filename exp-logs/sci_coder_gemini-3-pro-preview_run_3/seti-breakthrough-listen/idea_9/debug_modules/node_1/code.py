import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import SETIDataset, get_transforms
from library.model import SiameseEfficientNetV2
from library.engine import train_model, create_submission


def main():
    # 1. Setup and Configuration
    print("--- 1. Setup and Configuration ---")
    # Set seed for reproducibility
    seed_everything(42)

    # Initialize Config directories
    Config.setup()

    # Override Config for speed optimization in this demo
    print("Overriding configuration for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers for simple script

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading and Verification
    print("\n--- 2. Data Loading and Verification ---")

    # Load metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_CSV}")

    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Sample data for speed
    df_train_sub = df_train.sample(
        n=Config.DEBUG_SUBSET_SIZE, random_state=Config.SEED
    ).reset_index(drop=True)
    df_val_sub = df_val.sample(
        n=Config.DEBUG_SUBSET_SIZE, random_state=Config.SEED
    ).reset_index(drop=True)
    df_test_sub = df_test.sample(
        n=Config.DEBUG_SUBSET_SIZE, random_state=Config.SEED
    ).reset_index(drop=True)

    print(f"Training subset shape: {df_train_sub.shape}")

    # Instantiate Datasets
    train_dataset = SETIDataset(df_train_sub, transform=get_transforms("train"))
    val_dataset = SETIDataset(df_val_sub, transform=get_transforms("valid"))

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verification: Check batch shapes
    images, targets = next(iter(train_loader))

    # Expected Image Shape: (Batch, 6, 288, 256)
    # 6 channels because dataset reorders them for Siamese input (A,A,A,B,C,D)
    # 288 height because of padding in dataset.py
    expected_shape = (Config.BATCH_SIZE, 6, Config.IMG_HEIGHT, Config.IMG_WIDTH)

    if images.shape != expected_shape:
        raise AssertionError(
            f"Batch shape mismatch. Expected {expected_shape}, got {images.shape}"
        )

    if targets.shape[0] != Config.BATCH_SIZE:
        raise AssertionError(
            f"Target batch size mismatch. Expected {Config.BATCH_SIZE}, got {targets.shape[0]}"
        )

    print("Data loading logic verified. Batch shapes are correct.")

    # 3. Model Initialization and Forward Pass
    print("\n--- 3. Model Initialization and Forward Pass ---")

    model = SiameseEfficientNetV2(pretrained=True)
    model.to(device)

    # Run dummy forward pass
    # We use the batch fetched earlier
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Expected Output Shape: (Batch, 1) (Logits)
    if outputs.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {outputs.shape}"
        )

    print("Model forward pass verified.")

    # 4. Training Loop Demonstration
    print("\n--- 4. Training Loop Demonstration ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    # Run training using the engine
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=os.path.join(Config.WORK_DIR, "demo_best_model.pth"),
    )

    print(f"Training demonstration complete. Best AUC: {best_auc}")

    # Verify model file was saved
    model_path = os.path.join(Config.WORK_DIR, "demo_best_model.pth")
    if not os.path.exists(model_path):
        raise AssertionError("Model checkpoint was not saved.")

    print("Model checkpoint verified.")

    # 5. Inference and Submission
    print("\n--- 5. Inference and Submission ---")

    # Create test loader
    test_dataset = SETIDataset(df_test_sub, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load best model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate submission
    submission_path = os.path.join(Config.WORK_DIR, "demo_submission.csv")
    create_submission(
        model=model,
        loader=test_loader,
        device=device,
        test_df=df_test_sub,
        output_path=submission_path,
    )

    # Verify submission file
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    if df_sub.shape[0] != Config.DEBUG_SUBSET_SIZE:
        raise AssertionError(
            f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {df_sub.shape[0]}"
        )

    if list(df_sub.columns) != ["id", "target"]:
        raise AssertionError(
            f"Submission columns mismatch. Expected ['id', 'target'], got {list(df_sub.columns)}"
        )

    print("Submission file verified.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
