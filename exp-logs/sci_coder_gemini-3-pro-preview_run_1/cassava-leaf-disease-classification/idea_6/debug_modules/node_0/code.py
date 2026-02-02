import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import CFG
from library.utils import seed_everything, get_llrd_params
from library.dataset import get_dataloaders, rand_bbox
from library.model import CassavaModel
from library.engine import (
    SoftTargetCrossEntropy,
    train_one_epoch,
    valid_one_epoch,
    inference_fn,
)


def run_pipeline_demo():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print(">>> [1/7] Configuring Pipeline for Demo...")

    # Override CFG settings for speed and resource efficiency
    CFG.debug = True  # Use tiny subset (100 train, 50 val, 50 test)
    CFG.batch_size = 8  # Small batch size
    CFG.num_workers = 2  # Minimal workers
    CFG.img_size_p1 = 224  # Reduced image size (standard 224x224)
    CFG.epochs_base = 1  # Run only 1 epoch
    CFG.model_name = "convnext_tiny"  # Lightweight backbone for speed
    CFG.print_freq = 10  # Print logs frequently

    # Ensure directories exist
    CFG.setup()

    # Set random seeds for reproducibility
    seed_everything(CFG.seed)

    print(f"    Device: {CFG.device}")
    print(f"    Model: {CFG.model_name}")
    print(f"    Debug Mode: {CFG.debug}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n>>> [2/7] Verifying Data Pipeline...")

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        img_size=CFG.img_size_p1, debug=True
    )

    # Fetch one batch from training loader
    images, labels = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    # Assertions to verify data integrity
    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.img_size_p1,
        CFG.img_size_p1,
    ), "Incorrect image tensor shape"
    assert labels.shape == (CFG.batch_size,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.int64, "Labels should be int64"

    # Verify rand_bbox utility (used for CutMix)
    bbox = rand_bbox((CFG.img_size_p1, CFG.img_size_p1), lam=0.5)
    assert len(bbox) == 4, "rand_bbox must return 4 coordinates"
    x1, y1, x2, y2 = bbox
    assert x2 > x1 and y2 > y1, "Bounding box coordinates are invalid"
    print("    Data Pipeline verified successfully.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n>>> [3/7] Initializing Model...")

    # Initialize model (pretrained=False to avoid download time/errors)
    model = CassavaModel(model_name=CFG.model_name, pretrained=False)
    model.to(CFG.device)

    # Perform dummy forward pass
    dummy_input = images.to(CFG.device)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Model output shape mismatch"
    print("    Model initialized and forward pass verified.")

    # ==========================================
    # 4. Optimizer & Loss Configuration
    # ==========================================
    print("\n>>> [4/7] Configuring Optimizer & Loss...")

    # Setup LLRD (Layer-wise Learning Rate Decay) parameters
    # This verifies the logic in utils.py for parameter grouping
    optimizer_params = get_llrd_params(
        model,
        base_lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        decay_factor=CFG.llrd_decay,
    )

    # Assert we have parameter groups (ConvNeXt structure should trigger grouping)
    assert len(optimizer_params) > 0, "Optimizer parameter groups failed to generate"

    optimizer = torch.optim.AdamW(optimizer_params, lr=CFG.lr)
    loss_fn = SoftTargetCrossEntropy()

    # Verify Loss Calculation with dummy soft targets
    # We convert labels to one-hot because SoftTargetCrossEntropy expects probabilities
    dummy_targets = F.one_hot(labels.to(CFG.device), CFG.num_classes).float()
    initial_loss = loss_fn(logits, dummy_targets)

    print(f"    Initial Loss: {initial_loss.item():.4f}")
    assert not torch.isnan(initial_loss), "Loss is NaN"
    assert initial_loss.item() > 0, "Loss should be positive"
    print("    Optimizer and Loss configured successfully.")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n>>> [5/7] Running Training Loop (1 Epoch)...")

    # Run one epoch of training
    # Note: train_one_epoch handles MixUp/CutMix internally
    train_loss = train_one_epoch(
        epoch=0,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        device=CFG.device,
    )

    print(f"    Avg Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN"

    # ==========================================
    # 6. Validation Loop Execution
    # ==========================================
    print("\n>>> [6/7] Running Validation Loop...")

    # Run validation
    val_loss, val_acc = valid_one_epoch(
        epoch=0, model=model, loss_fn=loss_fn, val_loader=val_loader, device=CFG.device
    )

    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val Accuracy: {val_acc:.4f}")
    assert 0.0 <= val_acc <= 1.0, "Accuracy score is out of bounds [0, 1]"

    # ==========================================
    # 7. Inference & Submission Generation
    # ==========================================
    print("\n>>> [7/7] Running Inference & Generating Submission...")

    # Run inference (includes TTA if CFG.tta is True)
    predictions = inference_fn(model, test_loader, CFG.device)

    print(f"    Predictions Shape: {predictions.shape}")

    # In debug mode, test set is sampled to 50 images
    expected_test_size = 50
    assert predictions.shape == (
        expected_test_size,
        CFG.num_classes,
    ), "Prediction shape mismatch"

    # Generate Submission CSV
    # We reload the test CSV and sample it identically to match the debug loader's content
    df_test = pd.read_csv(CFG.test_csv)
    df_test = df_test.sample(n=expected_test_size, random_state=CFG.seed).reset_index(
        drop=True
    )

    # Assign predicted labels
    df_test["label"] = predictions.argmax(axis=1)

    # Save
    submission_path = os.path.join(CFG.working_dir, "demo_submission.csv")
    df_test[["image_id", "label"]].to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")

    # Final verification
    assert os.path.exists(submission_path), "Submission file missing"

    print("\n>>> Demo Run Completed Successfully.")


if __name__ == "__main__":
    run_pipeline_demo()
