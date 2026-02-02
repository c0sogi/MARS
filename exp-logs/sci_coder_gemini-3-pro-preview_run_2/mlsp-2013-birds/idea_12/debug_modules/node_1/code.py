import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure the current directory is in the path to import the library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_device
from library.data import BirdDataset, get_transforms, prepare_folds
from library.models import get_model
from library.engine import train_one_epoch, validate, SWAHandler, mixup_data, train_fold


def main():
    print("Starting Library Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Setup and Configuration")

    # Set reproducible seed
    set_seed(42)

    # Get compute device
    device = get_device()
    print(f"Device: {device}")

    # Override Config settings for a fast demonstration
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.SWA_START_EPOCH = 1  # Start SWA immediately in the 2nd epoch

    # Verify Metadata Existence
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train CSV not found at {Config.TRAIN_CSV}")
    if not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(f"Val CSV not found at {Config.VAL_CSV}")

    print("Configuration verified and adjusted for demo speed.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Data Pipeline Verification")

    # Load training metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    print(f"Total training samples available: {len(df_train)}")

    # Create a small subset (10 samples) for testing
    df_subset = df_train.head(10).copy()

    # Define transforms (using ResNet input size)
    h, w = Config.RESNET_INPUT_SIZE
    transform = get_transforms(h, w, phase="train")

    # Instantiate Dataset
    dataset = BirdDataset(df_subset, transform=transform, phase="train")
    print(f"Subset dataset created with {len(dataset)} samples.")

    # Verify __getitem__
    sample = dataset[0]
    image = sample["image"]
    label = sample["label"]
    rec_id = sample["rec_id"]

    print(f"  Sample Image Shape: {image.shape} (Expected: 3, {h}, {w})")
    print(f"  Sample Label Shape: {label.shape} (Expected: {Config.NUM_CLASSES})")

    # Assertions for correctness
    assert image.shape == (3, h, w), f"Image shape mismatch: {image.shape}"
    assert label.shape == (Config.NUM_CLASSES,), f"Label shape mismatch: {label.shape}"
    assert isinstance(rec_id, torch.Tensor), "rec_id must be a tensor"

    # Verify DataLoader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    batch = next(iter(loader))
    print(f"  Batch Image Shape: {batch['image'].shape}")
    print(f"  Batch Label Shape: {batch['label'].shape}")

    assert batch["image"].shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    # Verify Fold Preparation Logic
    # This function reads metadata and creates stratified folds
    print("  Testing prepare_folds()...")
    # We force generation (load_cached_data=False) to test the logic
    folds_df = prepare_folds(load_cached_data=False)
    assert "fold" in folds_df.columns, "Folds DataFrame missing 'fold' column"
    print(f"  Folds DataFrame generated with shape: {folds_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Model Architecture Verification")

    model_name = Config.MODEL_A_NAME  # resnet18
    print(f"Instantiating model: {model_name}")

    # Create model (pretrained=False to avoid downloading weights during demo)
    model = get_model(model_name, pretrained=False, device=device)

    # Verify Forward Pass
    dummy_input = torch.randn(Config.BATCH_SIZE, 3, h, w).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Engine Component Verification
    # -------------------------------------------------------------------------
    print("\n[4] Engine Component Verification")

    # Verify Mixup
    print("  Testing Mixup augmentation...")
    x = torch.randn(4, 3, h, w).to(device)
    y = torch.randint(0, 2, (4, Config.NUM_CLASSES)).float().to(device)
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4, device=device)

    assert mixed_x.shape == x.shape, "Mixup altered input shape"
    assert y_a.shape == y.shape, "Mixup altered label shape"
    print(f"  Mixup lambda: {lam:.4f}")

    # Verify SWA Handler
    print("  Testing SWAHandler...")
    swa_handler = SWAHandler(model)
    swa_handler.update(model)  # Accumulate weights
    swa_model = swa_handler.finalize(model)  # Average weights
    assert isinstance(swa_model, torch.nn.Module), "SWA finalize did not return a model"
    print("  SWAHandler update and finalize successful.")

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Integration Test
    # -------------------------------------------------------------------------
    print("\n[5] Full Training Loop Integration Test")
    print("  Running train_fold() simulation on subset data...")

    # We use the subset loader for both training and validation to verify the loop mechanics.
    # train_fold handles optimizer creation, loss calculation, SWA, and saving.

    try:
        # Use a dummy fold index for the demo
        demo_fold_idx = 99

        auc_score = train_fold(
            fold_idx=demo_fold_idx,
            model_name=model_name,
            train_loader=loader,
            val_loader=loader,
            device=device,
        )

        print(f"  Integration Test Completed. Final AUC: {auc_score:.4f}")

        # Verify model artifact was saved
        expected_model_path = os.path.join(
            Config.WORK_DIR, f"model_fold{demo_fold_idx}_{model_name}.pth"
        )
        if os.path.exists(expected_model_path):
            print(f"  Model artifact successfully saved at: {expected_model_path}")
        else:
            raise FileNotFoundError("Model artifact was not saved.")

    except Exception as e:
        print(f"  Integration Test Failed: {e}")
        raise e

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
