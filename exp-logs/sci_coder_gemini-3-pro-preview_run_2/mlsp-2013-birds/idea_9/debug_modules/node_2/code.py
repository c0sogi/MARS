import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the python path to allow imports from library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_pos_weights
from library.data import get_folds, BirdDataset, get_transforms, MixupCollate
from library.models import get_bird_model
from library.train import train_one_epoch, validate, run_training


def main():
    print("=== Bird Species Classification Pipeline Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("[1] Configuring parameters for rapid demonstration...")

    # Override Config constants to ensure the demo runs quickly (within seconds/minutes)
    Config.N_FOLDS = 2  # Run only 2 folds instead of 5
    Config.MODEL_ARCHS = ["resnet18"]  # Use only the smallest backbone
    Config.TOTAL_STEPS = 5  # Run only 5 steps per fold
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debugging
    Config.DEBUG = True

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("    Configuration updated: 2 Folds, 5 Steps, ResNet18, Batch Size 4.")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Generate/Load Folds
    # We force load_cached_data=False to test the generation logic
    df_folds = get_folds(load_cached_data=False)

    # Assertions for Dataframe
    assert isinstance(df_folds, pd.DataFrame), "get_folds must return a DataFrame"
    assert "fold" in df_folds.columns, "DataFrame must contain 'fold' column"
    assert len(df_folds) > 0, "DataFrame should not be empty"
    print(f"    Folds generated successfully. Total samples: {len(df_folds)}")

    # Instantiate Dataset (Train Mode)
    train_transform = get_transforms(mode="train")
    train_dataset = BirdDataset(df_folds, mode="train", transform=train_transform)

    # Check single item retrieval
    img, label = train_dataset[0]
    assert img.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Image shape mismatch. Expected {(3, Config.IMG_HEIGHT, Config.IMG_WIDTH)}, got {img.shape}"
    assert label.shape == (
        Config.NUM_CLASSES,
    ), f"Label shape mismatch. Expected {(Config.NUM_CLASSES,)}, got {label.shape}"
    assert isinstance(img, torch.Tensor), "Image must be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"
    print("    Single item retrieval verified.")

    # Instantiate DataLoader with Mixup
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MixupCollate(alpha=1.0),  # Force mixup
        num_workers=0,
    )

    # Check batch retrieval
    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Batch image shape mismatch: {batch_imgs.shape}"
    assert batch_labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Batch label shape mismatch: {batch_labels.shape}"
    print("    DataLoader and MixupCollate verified.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Initialize Model (ResNet18)
    # Using pretrained=False for speed in demo, though Config uses True
    model = get_bird_model("resnet18", pretrained=False)
    model.to(Config.DEVICE)

    # Perform Forward Pass
    with torch.no_grad():
        # Move batch to device
        inputs = batch_imgs.to(Config.DEVICE)
        logits = model(inputs)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"
    print("    Model instantiation and forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Training & Validation Logic Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Training and Validation Logic...")

    # Setup Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Calculate dummy pos_weights
    pos_weights = torch.ones(Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Test train_one_epoch
    print("    Running single training epoch...")
    loss, steps = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        Config.DEVICE,
        current_step=0,
        total_steps=2,  # Limit to 2 steps
    )

    assert isinstance(loss, float), "Training loss must be a float"
    assert not np.isnan(loss), "Training loss should not be NaN"
    assert steps > 0, "Steps taken should be greater than 0"
    print(f"    Training step successful. Loss: {loss:.4f}")

    # Test validate
    print("    Running validation...")
    val_loss, val_auc = validate(model, train_loader, criterion, Config.DEVICE)

    assert isinstance(val_loss, float), "Validation loss must be a float"
    assert isinstance(val_auc, float), "Validation AUC must be a float"
    # AUC might be 0.5 if batch size is small and classes are constant, which is handled
    assert 0.0 <= val_auc <= 1.0, "AUC must be between 0 and 1"
    print(f"    Validation successful. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # -------------------------------------------------------------------------
    # 5. Full Integration Test
    # -------------------------------------------------------------------------
    print("\n[5] Running Full Integration Test (run_training)...")
    print("    This will execute the training loop for 2 folds with 5 steps each.")

    try:
        run_training(load_cached_folds=True)
        print("    Integration test completed successfully.")
    except Exception as e:
        print(f"    Integration test failed with error: {e}")
        raise e

    # Check if model files were created
    expected_model_path = os.path.join(Config.WORKING_DIR, "model_resnet18_fold_0.pth")
    if os.path.exists(expected_model_path):
        print(f"    Verified: Model file created at {expected_model_path}")
    else:
        # It's possible no model is saved if validation AUC doesn't improve over 0.0 (unlikely but possible)
        # However, run_training prints when saving.
        print(
            f"    Note: Model file not found at {expected_model_path} (Check if AUC improved)."
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
