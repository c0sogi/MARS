import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Import library modules
from library.config import CFG
from library.utils import seed_everything, calculate_class_weights
from library.dataset import AppleDataset, get_transforms, prepare_folds
from library.model import AppleResNet34
from library.engine import train_one_epoch, valid_one_epoch
from library.workflow import (
    run_calibration_phase,
    run_production_phase,
    generate_submission,
)


def main():
    print("Starting Apple Disease Detection Demo...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Overriding Configuration for Demo Speed")
    # Reduce computational load for the demonstration
    CFG.calibration_epochs = 1  # Run only 1 epoch per fold
    CFG.n_folds = 2  # Run only 2 folds instead of 5
    CFG.ensemble_seeds = [42]  # Train only 1 model in production phase
    CFG.batch_size = 16  # Adjust batch size
    CFG.num_workers = 2  # Reduce workers

    # Ensure clean working directory for the demo
    if os.path.exists(CFG.working_dir):
        shutil.rmtree(CFG.working_dir)
    os.makedirs(CFG.working_dir, exist_ok=True)
    os.makedirs(CFG.models_dir, exist_ok=True)

    seed_everything(CFG.seed)

    # ==========================================
    # 2. Component Validation: Dataset & Transforms
    # ==========================================
    print("\n[2] Validating Dataset and Transforms")

    # Load metadata (using the library function)
    # Note: We force load_cached_data=False to ensure we test the creation logic
    df = prepare_folds(load_cached_data=False)

    # Check DataFrame structure
    assert "fold" in df.columns, "prepare_folds failed to create 'fold' column"
    assert len(df) > 0, "DataFrame is empty"
    print(f"   Data loaded successfully. Shape: {df.shape}")

    # Initialize Dataset
    train_ds = AppleDataset(df, transform=get_transforms("train"))

    # Fetch a sample
    img, label, img_id = train_ds[0]

    # Validations
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert img.shape == (
        3,
        CFG.img_size,
        CFG.img_size,
    ), f"Incorrect image shape: {img.shape}"
    assert label.shape == (4,), f"Incorrect label shape: {label.shape}"
    assert isinstance(img_id, str), "Image ID is not a string"
    print("   Dataset sample validation passed.")

    # ==========================================
    # 3. Component Validation: Model
    # ==========================================
    print("\n[3] Validating Model Architecture")

    model = AppleResNet34(
        pretrained=False
    )  # No need to download weights for shape check
    model.to(CFG.device)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, CFG.img_size, CFG.img_size).to(CFG.device)

    with torch.no_grad():
        output = model(dummy_input)

    # Validations
    assert output.shape == (
        2,
        4,
    ), f"Model output shape mismatch. Expected (2, 4), got {output.shape}"
    print("   Model forward pass validation passed.")

    # ==========================================
    # 4. Component Validation: Engine (Train/Valid Step)
    # ==========================================
    print("\n[4] Validating Training Engine")

    # Setup minimal components for a single step check
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Run one training step
    print("   Running single training epoch (subset)...")
    # We'll just run the function; it iterates the whole loader.
    # Since we want speed, let's artificially limit the loader for this specific check
    # by breaking the loop in a custom way or just trusting the small dataset size is fast enough.
    # Given the constraints, we will run the provided `train_one_epoch` on the full loader
    # but the loader is fast enough (1300 imgs).

    loss = train_one_epoch(model, loader, criterion, optimizer, None, CFG.device)
    assert not np.isnan(loss), "Training loss is NaN"
    print(f"   Train step successful. Loss: {loss:.4f}")

    # Run one validation step
    print("   Running single validation epoch...")
    val_loss, val_preds, val_labels = valid_one_epoch(
        model, loader, criterion, CFG.device
    )
    assert val_preds.shape == (len(df), 4), "Validation predictions shape mismatch"
    print(f"   Valid step successful. Loss: {val_loss:.4f}")

    # ==========================================
    # 5. Workflow Execution: Phase 1 (Calibration)
    # ==========================================
    print("\n[5] Executing Phase 1: Proxy Calibration")
    # This runs 5-fold CV (reduced to 2 folds via CFG override) for 1 epoch each

    optimal_epoch = run_calibration_phase(load_cached_data=True)

    assert isinstance(optimal_epoch, int), "Optimal epoch must be an integer"
    assert optimal_epoch > 0, "Optimal epoch must be positive"
    print(f"   Phase 1 complete. Optimal Epoch: {optimal_epoch}")

    # ==========================================
    # 6. Workflow Execution: Phase 2 (Production)
    # ==========================================
    print("\n[6] Executing Phase 2: Production Training")
    # Trains on full data using optimal_epoch

    run_production_phase(optimal_epoch, load_cached_data=True)

    # Verify model file creation
    expected_model_path = os.path.join(
        CFG.models_dir, f"resnet34_seed_{CFG.ensemble_seeds[0]}.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"   Phase 2 complete. Model saved to {expected_model_path}")

    # ==========================================
    # 7. Workflow Execution: Inference
    # ==========================================
    print("\n[7] Executing Inference & Submission")

    generate_submission()

    # Verify submission file
    assert os.path.exists(CFG.submission_path), "Submission file not found"

    sub_df = pd.read_csv(CFG.submission_path)
    test_meta = pd.read_csv(CFG.test_metadata_path)

    assert len(sub_df) == len(test_meta), "Submission row count mismatch"
    assert (
        list(sub_df.columns) == ["image_id"] + CFG.target_cols
    ), "Submission columns mismatch"

    # Verify probabilities sum to ~1 (Softmax applied)
    # Note: Floating point precision might make it slightly off 1.0
    row_sums = sub_df[CFG.target_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print(f"   Submission generated successfully at {CFG.submission_path}")
    print("\nDemo Completed Successfully!")


if __name__ == "__main__":
    main()
