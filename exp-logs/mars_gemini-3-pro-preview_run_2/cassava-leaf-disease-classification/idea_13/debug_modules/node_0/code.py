import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from timm.data import Mixup

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import load_metadata, CassavaDataset, get_transforms
from library.model import CassavaModel, ModelEMA
from library.losses import CassavaLoss
from library.engine import train_one_epoch, validate
from library.inference import predict_test_set


def run_demo():
    # 1. Setup and Configuration
    print(">>> Setting up configuration...")
    seed_everything(42)

    # Override Config for fast execution
    Config.DEBUG = True
    Config.PHASE_1_EPOCHS = 1
    Config.PHASE_2_EPOCHS = 0  # Skip phase 2 for demo
    Config.NUM_FOLDS = 1  # Only run 1 fold logic
    Config.PHASE_1_IMG_SIZE = 112  # Small size for speed
    Config.PHASE_1_BATCH_SIZE = 8

    # Re-run setup to ensure directories exist (though import does this)
    Config.setup()

    logger = get_logger()
    logger.info("Configuration overrides applied for demo speed.")

    # 2. Data Pipeline Verification
    print("\n>>> Verifying Data Pipeline...")

    # Load metadata (debug=True loads a small subset)
    df_train = load_metadata("train", debug=True)
    df_val = load_metadata("val", debug=True)

    assert len(df_train) > 0, "Train metadata is empty"
    assert len(df_val) > 0, "Val metadata is empty"

    # Create Datasets
    train_transform = get_transforms("train", img_size=Config.PHASE_1_IMG_SIZE)
    val_transform = get_transforms("val", img_size=Config.PHASE_1_IMG_SIZE)

    train_dataset = CassavaDataset(
        df_train, transforms=train_transform, output_label=True
    )
    val_dataset = CassavaDataset(df_val, transforms=val_transform, output_label=True)

    # Verify single item
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert img.shape == (
        3,
        Config.PHASE_1_IMG_SIZE,
        Config.PHASE_1_IMG_SIZE,
    ), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.PHASE_1_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.PHASE_1_BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Data Pipeline verified successfully.")

    # 3. Model and Loss Verification
    print("\n>>> Verifying Model and Loss...")

    device = Config.DEVICE
    # Use pretrained=False to avoid downloading weights during demo
    model = CassavaModel(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(
        2, 3, Config.PHASE_1_IMG_SIZE, Config.PHASE_1_IMG_SIZE
    ).to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected (2, 5), got {logits.shape}"

    # Verify EMA
    model_ema = ModelEMA(model, device=device)
    assert model_ema.module is not model, "EMA module should be a copy"

    # Verify Loss
    criterion = CassavaLoss(smoothing=0.1)

    # Case A: Hard targets (Standard CrossEntropy)
    targets_hard = torch.tensor([0, 1]).to(device)
    loss_hard = criterion(logits, targets_hard)
    assert loss_hard.ndim == 0, "Loss should be a scalar"

    # Case B: Soft targets (MixUp)
    targets_soft = torch.zeros_like(logits).to(device)
    targets_soft[0, 0] = 1.0
    targets_soft[1, 1] = 1.0
    loss_soft = criterion(logits, targets_soft)
    assert loss_soft.ndim == 0, "Loss should be a scalar"

    print("Model and Loss logic verified.")

    # 4. Training Loop Execution
    print("\n>>> Executing Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Setup Mixup for Phase 1
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=Config.NUM_CLASSES,
    )

    # Run Train Step
    avg_loss = train_one_epoch(
        epoch=0,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_ema=model_ema,
        mixup_fn=mixup_fn,
        accumulation_steps=1,
    )

    print(f"Training epoch completed. Avg Loss: {avg_loss:.4f}")

    # Run Validation Step
    acc, val_loss = validate(
        model=model, val_loader=val_loader, criterion=criterion, device=device
    )

    print(f"Validation completed. Acc: {acc:.2f}, Loss: {val_loss:.4f}")

    # 5. Inference and Submission Verification
    print("\n>>> Verifying Inference Pipeline...")

    # Save dummy checkpoints for all folds (Config.NUM_FOLDS is set to 1, but inference loops 5 in original config)
    # We need to ensure the inference script finds what it looks for.
    # The inference script loops range(Config.NUM_FOLDS).
    # Since we modified Config.NUM_FOLDS to 1, it will look for fold 0.

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_fold_0.pth")
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    print(f"Saved dummy checkpoint to {checkpoint_path}")

    # Run Inference
    # predict_test_set uses Config.NUM_FOLDS. We set it to 1, so it runs fast.
    predict_test_set(debug=True)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    assert "image_id" in df_sub.columns, "Missing image_id column"
    assert "label" in df_sub.columns, "Missing label column"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference pipeline verified successfully.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
