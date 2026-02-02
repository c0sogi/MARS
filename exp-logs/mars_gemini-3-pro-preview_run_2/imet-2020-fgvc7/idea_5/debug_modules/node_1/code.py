import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import load_dataset, ArtworkDataset
from library.models import get_model, ModelEMA
from library.losses import AsymmetricLoss, DistillationLoss
from library.engine import (
    train_one_epoch,
    validate,
    generate_soft_labels,
    inference,
    find_best_threshold,
    create_submission,
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("1. Initializing Configuration and Environment...")

    # Override Config for a fast, lightweight demo
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 images
    Config.IMG_SIZE = 224  # Smaller image size for speed
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.EPOCHS = 1

    # Use a lightweight model for demonstration (avoids loading large weights)
    Config.STUDENT_MODEL = "resnet18"
    Config.TEACHER_MODEL_1 = "resnet18"

    # Setup working directory for demo outputs
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to demo directory
    Config.TEACHER_PREDS_PATH = os.path.join(Config.WORKING_DIR, "val_logits.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Set seeds
    seed_everything(Config.SEED)
    device = get_device()
    print(f"   Device: {device}")
    print(f"   Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Dataset Validation
    # -------------------------------------------------------------------------
    print("\n2. Verifying Dataset Loading...")

    # Load training dataset (Hard labels only initially)
    train_ds = load_dataset("train", debug=True, use_soft_labels=False)

    # Assertions
    assert (
        len(train_ds) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset size mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(train_ds)}"

    # Check item structure
    img, target = train_ds[0]
    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Target shape mismatch. Expected ({Config.NUM_CLASSES},), got {target.shape}"

    print(f"   Train Dataset loaded. Size: {len(train_ds)}")
    print(f"   Image Shape: {img.shape}, Target Shape: {target.shape}")

    # -------------------------------------------------------------------------
    # 3. Model & EMA Initialization
    # -------------------------------------------------------------------------
    print("\n3. Initializing Models...")

    # Create Student Model
    # pretrained=False to avoid downloading weights during demo
    model = get_model(
        Config.STUDENT_MODEL, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(device)

    # Create EMA Model
    ema_model = ModelEMA(model)

    # Verify EMA
    assert hasattr(ema_model, "ema"), "EMA model not initialized correctly"
    print("   Student model and EMA initialized.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n4. Verifying Loss Functions...")

    # Dummy data
    batch_size = 4
    dummy_logits = torch.randn(batch_size, Config.NUM_CLASSES).to(device)
    dummy_targets = (
        torch.randint(0, 2, (batch_size, Config.NUM_CLASSES)).float().to(device)
    )

    # Test Asymmetric Loss
    asl = AsymmetricLoss()
    loss_val = asl(dummy_logits, dummy_targets)
    assert not torch.isnan(loss_val), "ASL Loss returned NaN"
    assert loss_val.item() > 0, "ASL Loss should be positive"
    print(f"   ASL Loss Check Passed: {loss_val.item():.4f}")

    # Test Distillation Loss
    distill_loss = DistillationLoss()
    dummy_teacher_logits = torch.randn(batch_size, Config.NUM_CLASSES).to(device)
    d_loss_val = distill_loss(dummy_logits, dummy_teacher_logits, dummy_targets)
    assert not torch.isnan(d_loss_val), "Distillation Loss returned NaN"
    print(f"   Distillation Loss Check Passed: {d_loss_val.item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Soft Label Generation (Teacher Simulation)
    # -------------------------------------------------------------------------
    print("\n5. Generating Soft Labels (Teacher Simulation)...")

    # Create a dummy teacher model
    teacher_model = get_model(
        Config.TEACHER_MODEL_1, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    teacher_model.to(device)

    # Create a loader for generation (deterministic transforms internally handled by dataset if val mode used,
    # but generate_soft_labels usually runs on train set. We use the train_ds we already loaded)
    # Note: For generation, we usually want deterministic transforms, but for this demo, we use the existing ds.
    gen_loader = DataLoader(
        train_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Generate
    soft_logits = generate_soft_labels(
        models=[teacher_model],
        loader=gen_loader,
        device=device,
        save_path=Config.TEACHER_PREDS_PATH,
        load_cached_data=False,  # Force generation
    )

    assert os.path.exists(Config.TEACHER_PREDS_PATH), "Soft labels file not saved"
    assert soft_logits.shape == (
        len(train_ds),
        Config.NUM_CLASSES,
    ), "Soft labels shape mismatch"
    print("   Soft labels generated and saved.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n6. Executing Training Loop...")

    # Reload dataset with soft labels
    train_ds_distill = load_dataset("train", debug=True, use_soft_labels=True)
    train_loader = DataLoader(
        train_ds_distill,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Run one epoch
    train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        device=device,
        epoch=1,
        use_ema=True,
        ema_model=ema_model,
    )
    print("   Training epoch completed.")

    # -------------------------------------------------------------------------
    # 7. Validation & Metrics
    # -------------------------------------------------------------------------
    print("\n7. Validating...")

    val_ds = load_dataset("val", debug=True)
    val_loader = DataLoader(
        val_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=2
    )

    val_loss, val_preds, val_targets = validate(model, val_loader, device)

    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Predictions Shape: {val_preds.shape}")

    # Find best threshold
    # Since model is random/untrained, we mock probabilities to ensure threshold finding works
    # (Real predictions might be all very low, causing 0.0 score everywhere)
    # But let's try with real output first.
    best_thresh = find_best_threshold(
        val_targets, 1 / (1 + np.exp(-val_preds))
    )  # Sigmoid applied manually for metric

    # Save thresholds for inference
    np.save(os.path.join(Config.WORKING_DIR, "best_thresholds.npy"), best_thresh)

    # -------------------------------------------------------------------------
    # 8. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n8. Running Inference and Generating Submission...")

    test_ds = load_dataset("test", debug=True)
    test_loader = DataLoader(
        test_ds, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=2
    )

    ids, test_probs = inference(model, test_loader, device, use_tta=True)

    assert len(ids) == len(test_ds), "Number of test IDs mismatch"
    assert test_probs.shape == (
        len(test_ds),
        Config.NUM_CLASSES,
    ), "Test probabilities shape mismatch"

    create_submission(
        ids, test_probs, threshold=best_thresh, save_path=Config.SUBMISSION_PATH
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "attribute_ids" in df_sub.columns
    ), "Submission columns missing"
    print(f"   Submission generated at {Config.SUBMISSION_PATH}")
    print(f"   Rows: {len(df_sub)}")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
