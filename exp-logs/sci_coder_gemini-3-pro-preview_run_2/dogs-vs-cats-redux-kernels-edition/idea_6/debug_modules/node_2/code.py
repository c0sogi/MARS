import os
import sys
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import DogCatDataset, get_train_transforms, get_valid_transforms
from library.models import build_model
from library.engine import train_one_epoch, validate
from library.inference import predict_with_tta


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("[1] Setting up environment...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for the subset
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Clean and create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Subsets)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing data subsets...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_CSV}")

    train_df_full = pd.read_csv(Config.TRAIN_CSV)
    val_df_full = pd.read_csv(Config.VAL_CSV)
    test_df_full = pd.read_csv(Config.TEST_CSV)

    # Create small subsets to ensure quick execution
    train_subset = train_df_full.head(16).copy()
    val_subset = val_df_full.head(8).copy()
    test_subset = test_df_full.head(8).copy()

    print(f"    Train Subset: {len(train_subset)} samples")
    print(f"    Val Subset:   {len(val_subset)} samples")
    print(f"    Test Subset:  {len(test_subset)} samples")

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Datasets and DataLoaders...")

    # Initialize Datasets
    train_ds = DogCatDataset(
        train_subset, transforms=get_train_transforms(Config.IMG_SIZE), mode="train"
    )
    val_ds = DogCatDataset(
        val_subset, transforms=get_valid_transforms(Config.IMG_SIZE), mode="val"
    )
    test_ds = DogCatDataset(
        test_subset, transforms=get_valid_transforms(Config.IMG_SIZE), mode="test"
    )

    # Verify Dataset Output
    sample_img, sample_label = train_ds[0]
    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"
    print("    Dataset verification passed.")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # -------------------------------------------------------------------------
    # 4. Model Building
    # -------------------------------------------------------------------------
    print("\n[4] Building Model...")

    # Use the first architecture defined in Config
    # pretrained=False to avoid downloading large weights during this demo
    model_arch = Config.MODEL_ARCHS[0]
    print(f"    Architecture: {model_arch}")

    model = build_model(model_arch, pretrained=False, num_classes=1)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("    Model structure and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run training for one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )

    print(f"    Training completed. Average Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"

    # -------------------------------------------------------------------------
    # 6. Validation
    # -------------------------------------------------------------------------
    print("\n[6] Executing Validation...")

    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"    Validation completed. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    assert not np.isnan(val_loss), "Validation loss returned NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of bounds"

    # -------------------------------------------------------------------------
    # 7. Checkpointing
    # -------------------------------------------------------------------------
    print("\n[7] Testing Checkpoint Save/Load...")

    ckpt_filename = os.path.join(Config.WORKING_DIR, "demo_checkpoint.pth")

    # Save
    save_checkpoint(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_filename,
    )

    assert os.path.exists(ckpt_filename), "Checkpoint file was not created"

    # Load into a fresh model
    new_model = build_model(model_arch, pretrained=False, num_classes=1)
    new_model.to(device)

    checkpoint = load_checkpoint(ckpt_filename, new_model, device=device)
    assert "state_dict" in checkpoint, "Loaded checkpoint missing state_dict"
    print(f"    Checkpoint successfully saved to and loaded from {ckpt_filename}")

    # -------------------------------------------------------------------------
    # 8. Inference (Test Time Augmentation)
    # -------------------------------------------------------------------------
    print("\n[8] Running Inference with TTA...")

    # Use the loaded model for inference
    preds_dict = predict_with_tta(new_model, test_loader, device)

    # Verify predictions
    assert len(preds_dict) == len(
        test_subset
    ), f"Expected {len(test_subset)} predictions, got {len(preds_dict)}"

    first_id = test_subset.iloc[0]["id"]
    assert first_id in preds_dict, f"ID {first_id} missing from predictions"
    assert (
        0.0 <= preds_dict[first_id] <= 1.0
    ), "Probability value out of valid range [0, 1]"

    print("    Inference successful. Predictions generated.")

    # -------------------------------------------------------------------------
    # 9. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[9] Generating Submission File...")

    submission_rows = [{"id": int(k), "label": v} for k, v in preds_dict.items()]
    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df.sort_values("id")

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
