import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.modeling import get_model, verify_initialization
from library.workflow import AppleDiseaseWorkflow


def run_demo():
    print("Starting Apple Disease Detection Demo...")

    # ==========================================
    # 1. Configuration & Setup for Demo
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.IDEA_NAME = "demo_execution"
    Config.WORKING_DIR = f"./working/{Config.IDEA_NAME}"
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "output")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute requirements
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.N_FOLDS = 2  # Use only 2 folds for CV
    Config.PHASE2_SEEDS = [42]  # Use single seed for production training
    Config.IMG_SIZE = 128  # Smaller images for faster processing
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.NUM_WORKERS = 2  # Moderate workers

    # Seed everything
    seed_everything(Config.SEED)
    print("Configuration updated for speed.")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    # Get dataloaders for Fold 0
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        fold_idx=0, phase="phase1", batch_size=Config.BATCH_SIZE
    )

    # Check Class Weights
    print(f"Class Weights: {class_weights}")
    assert isinstance(class_weights, torch.Tensor), "Class weights should be a Tensor"
    assert (
        class_weights.shape[0] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} weights"

    # Verify Train Batch
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    expected_lbl_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        labels.shape == expected_lbl_shape
    ), f"Label shape mismatch. Expected {expected_lbl_shape}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print("Data Pipeline verified successfully.")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\n[3] Verifying Model Logic...")

    device = Config.DEVICE
    model = get_model(
        device=device, pretrained=False
    )  # Pretrained=False for speed in loading, logic remains same

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)

    # 3.1 Verify Initialization Logic
    # This checks if the custom head initialization produces low initial loss
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    verify_initialization(model, train_loader, criterion, device)

    # 3.2 Verify Forward Pass
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape is incorrect"

    # Check for NaNs
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    print("Model Logic verified successfully.")

    # ==========================================
    # 4. Workflow Execution
    # ==========================================
    print("\n[4] Executing Full Workflow...")

    workflow = AppleDiseaseWorkflow()

    # Phase 1: Calibration (CV)
    # This will run for Config.EPOCHS (1) over Config.N_FOLDS (2)
    print(">>> Running Phase 1...")
    optimal_epoch, use_tta = workflow.run_phase1()

    print(f"Phase 1 Complete. Optimal Epoch: {optimal_epoch}, Use TTA: {use_tta}")
    assert (
        isinstance(optimal_epoch, int) and optimal_epoch > 0
    ), "Optimal epoch must be a positive integer"
    assert isinstance(use_tta, bool), "use_tta must be boolean"

    # Phase 2: Production Training
    # This will run for optimal_epoch (1) using Config.PHASE2_SEEDS ([42])
    print(">>> Running Phase 2...")
    workflow.run_phase2(optimal_epoch)

    # Check if model file was created
    model_path = os.path.join(
        workflow.models_dir, f"phase2_model_seed_{Config.PHASE2_SEEDS[0]}.pth"
    )
    assert os.path.exists(model_path), f"Phase 2 model not found at {model_path}"
    print("Phase 2 Complete.")

    # Submission Generation
    print(">>> Generating Submission...")
    workflow.generate_submission(use_tta)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print("Workflow execution successful.")

    # ==========================================
    # 5. Submission Validation
    # ==========================================
    print("\n[5] Validating Submission Format...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(df_sub.head())

    # Check columns
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"

    # Check row count (Test set has 183 images)
    expected_rows = 183
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check value ranges (probabilities)
    # Note: Since we use Softmax in predict(), values should sum to ~1 per row
    probs = df_sub[["healthy", "multiple_diseases", "rust", "scab"]].values
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    print("Submission format valid.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
