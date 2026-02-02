import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library modules
from library.config import Config
from library.utils import seed_everything, get_class_weights, mixup_data
from library.data_loader import get_loaders, get_test_loader, load_full_train_data
from library.model_factory import get_model
from library.loss_factory import get_loss
from library.trainer import run_fold


def main():
    print("=== Starting Apple Disease Detection Pipeline Verification ===\n")

    # 1. Configuration Setup
    # We use debug=True to load a small subset of data for speed.
    # We set epochs=1 to ensure the training loop finishes quickly.
    print("Step 1: Initializing Configuration")
    config = Config(debug=True, epochs=1, batch_size=4)

    # Override working directory to a temporary location for this demo
    config.working_dir = "./working/demo_verification"
    os.makedirs(config.working_dir, exist_ok=True)

    print(f"  Debug Mode: {config.debug}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Working Directory: {config.working_dir}")

    # 2. Verify Utilities
    print("\nStep 2: Verifying Utilities (library.utils)")

    # Test Reproducibility
    seed_everything(config.seed)
    rand_a = np.random.rand()
    seed_everything(config.seed)
    rand_b = np.random.rand()
    assert (
        rand_a == rand_b
    ), "Seed everything failed: Random numbers are not reproducible."
    print("  [Pass] Seed reproducibility verified.")

    # Test Mixup Augmentation
    dummy_images = torch.randn(4, 3, 256, 256)
    dummy_labels = torch.randn(4, 4)  # Soft labels
    mixed_imgs, y_a, y_b, lam = mixup_data(dummy_images, dummy_labels, alpha=0.4)

    assert mixed_imgs.shape == dummy_images.shape, "Mixup: Image shape mismatch."
    assert y_a.shape == dummy_labels.shape, "Mixup: Label A shape mismatch."
    assert 0.0 <= lam <= 1.0, "Mixup: Lambda coefficient out of range [0, 1]."
    print("  [Pass] Mixup augmentation logic verified.")

    # 3. Verify Data Loading
    print("\nStep 3: Verifying Data Loading (library.data_loader)")

    # Test loading the full metadata
    full_df = load_full_train_data(config)
    assert not full_df.empty, "Data Loader: Full training dataframe is empty."
    # In debug mode, we expect a small sample (e.g., 100 rows)
    print(f"  Loaded {len(full_df)} rows (Debug Mode).")

    # Test Class Weights Calculation
    # We use the loaded df to calculate weights
    class_weights = get_class_weights(full_df, config.target_cols)
    assert (
        class_weights.shape[0] == config.num_classes
    ), "Class Weights: Incorrect number of classes."
    assert torch.is_tensor(class_weights), "Class Weights: Output is not a tensor."
    print("  [Pass] Class weights calculation verified.")

    # Test Train/Val Loaders
    train_loader, val_loader = get_loaders(fold=0, config=config)

    # Fetch one batch from train loader
    tr_images, tr_labels = next(iter(train_loader))
    assert tr_images.shape == (
        config.batch_size,
        3,
        config.img_size,
        config.img_size,
    ), f"Train Loader: Image batch shape mismatch. Got {tr_images.shape}"
    assert tr_labels.shape == (
        config.batch_size,
        config.num_classes,
    ), f"Train Loader: Label batch shape mismatch. Got {tr_labels.shape}"
    print("  [Pass] Train DataLoader verified.")

    # Test Test Loader
    test_loader, test_df = get_test_loader(config)
    te_images, _ = next(iter(test_loader))
    assert te_images.shape == (
        config.batch_size,
        3,
        config.img_size,
        config.img_size,
    ), "Test Loader: Image batch shape mismatch."
    print("  [Pass] Test DataLoader verified.")

    # 4. Verify Model Architecture
    print("\nStep 4: Verifying Model Architecture (library.model_factory)")
    model = get_model(config)

    # Move model to CPU for this quick check to avoid GPU overhead if not needed,
    # though config.device handles it. We'll stick to config.device.
    model.to(config.device)
    tr_images = tr_images.to(config.device)

    model.eval()
    with torch.no_grad():
        logits = model(tr_images)

    assert logits.shape == (
        config.batch_size,
        config.num_classes,
    ), f"Model: Output shape mismatch. Expected {(config.batch_size, config.num_classes)}, got {logits.shape}"
    print(f"  [Pass] Model forward pass verified (Output shape: {logits.shape}).")

    # 5. Verify Loss Function
    print("\nStep 5: Verifying Loss Function (library.loss_factory)")
    criterion = get_loss(config, class_weights=class_weights)

    # Simulate mixup outputs
    y_a_dummy = (
        torch.randn(config.batch_size, config.num_classes)
        .to(config.device)
        .softmax(dim=1)
    )
    y_b_dummy = (
        torch.randn(config.batch_size, config.num_classes)
        .to(config.device)
        .softmax(dim=1)
    )
    lam_dummy = 0.5

    loss_val = criterion(logits, y_a_dummy, y_b_dummy, lam_dummy)
    assert loss_val.dim() == 0, "Loss: Should return a scalar."
    assert not torch.isnan(loss_val), "Loss: Returned NaN."
    print(f"  [Pass] Loss calculation verified (Value: {loss_val.item():.4f}).")

    # 6. Verify Training Loop (Integration)
    print("\nStep 6: Verifying Trainer Integration (library.trainer)")
    print("  Starting training for 1 epoch on debug subset (Fold 0)...")

    # This runs the full training loop for one fold
    # Since debug=True and epochs=1, this should be very fast
    best_auc = run_fold(fold=0, config=config)

    assert isinstance(
        best_auc, float
    ), "Trainer: run_fold did not return a float score."
    print(f"  [Pass] Training loop completed. Best Val AUC: {best_auc:.4f}")

    # Check if model was saved
    expected_model_path = os.path.join(
        config.working_dir, f"{config.model_name}_fold_0.pth"
    )
    if best_auc > -1.0:  # Logic in trainer saves if val_auc > -1.0 (initial best)
        assert os.path.exists(
            expected_model_path
        ), f"Trainer: Model file not found at {expected_model_path}"
        print(f"  [Pass] Model checkpoint saved successfully.")
    else:
        print(
            "  [Info] No model saved (Validation AUC did not improve, which is unexpected but possible in edge cases)."
        )

    print("\n=== Verification Complete: All systems operational ===")


if __name__ == "__main__":
    main()
