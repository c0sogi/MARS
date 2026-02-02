import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import seed_everything, save_checkpoint, AverageMeter
from library.data import get_dataloaders, Mixup
from library.model import create_model, ModelEMA
from library.engine import train_one_epoch, validate
from library.inference import run_inference


def run_demo():
    print("Starting Cassava Leaf Disease Classification Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.LOG_DIR = os.path.join(Config.WORKING_DIR, "logs")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # Set demo parameters
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.PHASE_1["epochs"] = 1
    Config.PHASE_1["batch_size"] = 8
    Config.PHASE_1["image_size"] = 224
    Config.PHASE_2["batch_size"] = 8
    Config.PHASE_2["image_size"] = 224  # Keep small for demo

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Pipeline Verification
    # =========================================================================
    print("\n[2] Verifying Data Pipeline...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Subset for demo
    train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)
    val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)
    # Test df is handled by inference logic, but we'll inspect it

    print(f"Demo Train Size: {len(train_df)}")
    print(f"Demo Val Size: {len(val_df)}")

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, Config.PHASE_1
    )

    # Fetch one batch to verify
    images, labels = next(iter(train_loader))

    # Assertions
    assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    assert images.shape[2] == Config.PHASE_1["image_size"], "Image height mismatch"
    assert (
        labels.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and labels"

    print("Data loading successful. Batch shape:", images.shape)

    # =========================================================================
    # 3. Augmentation Logic (Mixup) Verification
    # =========================================================================
    print("\n[3] Verifying Mixup/CutMix Logic...")

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, num_classes=Config.NUM_CLASSES
    )

    # Create dummy batch
    dummy_imgs = torch.randn(8, 3, 224, 224)
    dummy_targets = torch.randint(0, 5, (8,))

    mixed_imgs, mixed_targets = mixup_fn(dummy_imgs, dummy_targets)

    # Assertions
    assert mixed_imgs.shape == dummy_imgs.shape, "Mixup altered image dimensions"
    assert mixed_targets.shape == (
        8,
        5,
    ), f"Expected soft targets (N, C), got {mixed_targets.shape}"
    assert torch.allclose(
        mixed_targets.sum(dim=1), torch.ones(8)
    ), "Soft targets do not sum to 1"

    print("Mixup application verified.")

    # =========================================================================
    # 4. Model Initialization & Forward Pass
    # =========================================================================
    print("\n[4] Initializing Model and EMA...")

    # Create model (pretrained=False to avoid downloading large weights during demo)
    model = create_model(pretrained=False)
    model.to(device)

    # Create EMA model
    model_ema = ModelEMA(model, device=device)

    # Forward pass check
    with torch.no_grad():
        output = model(images.to(device))

    assert output.shape == (
        images.shape[0],
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model forward pass successful.")

    # =========================================================================
    # 5. Training Loop Execution
    # =========================================================================
    print("\n[5] Executing Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.PHASE_1["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # Train one epoch
    train_loss, train_acc = train_one_epoch(
        epoch=0,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        mixup_fn=mixup_fn,
        model_ema=model_ema,
    )

    assert not np.isnan(train_loss), "Training loss is NaN"
    print(f"Training complete. Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_acc = validate(model, val_loader, device, loss_fn)
    print(f"Validation complete. Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("\n[6] Running Inference Pipeline...")

    # Save the "trained" model as a checkpoint for fold 0
    # The inference script looks for 'best_model_fold_0.pth'
    save_checkpoint(
        state=model.state_dict(),
        is_best=True,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        fold_idx=0,
    )

    # Also save dummy checkpoints for other folds to satisfy ensemble logic if needed,
    # but the inference script handles missing folds gracefully by skipping them.
    # We will just use fold 0.

    # Run inference
    # We use debug=True to limit the test set size as well
    submission_df = run_inference(
        checkpoint_dir=Config.CHECKPOINT_DIR,
        output_dir=Config.SUBMISSION_DIR,
        load_cached_data=False,
        debug=True,
        debug_size=20,
    )

    # Verify submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    loaded_sub = pd.read_csv(submission_path)
    assert (
        "image_id" in loaded_sub.columns and "label" in loaded_sub.columns
    ), "Submission columns missing"
    assert len(loaded_sub) > 0, "Submission file is empty"

    print(f"Inference successful. Submission generated with {len(loaded_sub)} rows.")
    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    run_demo()
