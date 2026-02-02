import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.dataset import get_dataloaders
from library.loss import AsymmetricLoss
from library.models import ArtworkClassifier, train_one_epoch, validate, inference


def main():
    print("=== Artwork Attribute Labeling Pipeline Demonstration ===\n")

    # 1. Configuration Setup
    # Override Config for a fast demonstration run
    print("[1] Configuring environment...")
    Config.debug = True  # Use small subset of data
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 2  # Reduce workers for demo
    Config.working_dir = "./working/demo_run"  # Separate demo directory

    # Create working directory
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Working Directory: {Config.working_dir}")

    # 2. Data Loading
    print("\n[2] Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers, debug=Config.debug
    )

    # Verify Train Batch
    images, targets = next(iter(train_loader))
    print(f"    Train Batch - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for data integrity
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Train image shape mismatch"
    assert targets.shape == (
        Config.batch_size,
        Config.num_classes,
    ), "Train target shape mismatch"
    assert targets.dtype == torch.float32, "Target dtype should be float32"

    # Verify Test Batch
    test_images, test_ids = next(iter(test_loader))
    print(f"    Test Batch  - Images: {test_images.shape}, IDs: {len(test_ids)}")
    assert len(test_ids) == Config.batch_size, "Test batch size mismatch"

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    # Use the first model in the config list
    model_name = Config.model_names[0]
    print(f"    Architecture: {model_name}")

    # Initialize model (pretrained=False for speed/offline safety in demo)
    model = ArtworkClassifier(model_name, Config.num_classes, pretrained=False).to(
        device
    )

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"    Output Logits Shape: {logits.shape}")
    assert logits.shape == (2, Config.num_classes), "Model output shape mismatch"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function...")
    criterion = AsymmetricLoss()
    dummy_targets = torch.randint(0, 2, (2, Config.num_classes)).float().to(device)

    loss = criterion(logits, dummy_targets)
    print(f"    Calculated Loss: {loss.item():.6f}")

    assert loss.item() >= 0, "Loss must be non-negative"
    assert not torch.isnan(loss), "Loss must not be NaN"

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        steps_per_epoch=len(train_loader),
        epochs=Config.epochs,
    )
    scaler = GradScaler()

    # Train for one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, device, scaler, ema=None
    )
    print(f"    Epoch 1 Train Loss: {train_loss:.6f}")
    assert train_loss > 0, "Train loss should be positive"

    # 6. Validation
    print("\n[6] Running Validation...")
    val_loss, val_f1, val_probs, val_targets = validate(
        model, val_loader, criterion, device
    )

    print(f"    Val Loss: {val_loss:.6f}")
    print(f"    Val F1 (default thresh): {val_f1:.6f}")
    print(f"    Val Probs Shape: {val_probs.shape}")

    # Assertions for validation outputs
    assert val_probs.shape == (
        len(val_loader.dataset),
        Config.num_classes,
    ), "Val probability shape mismatch"
    assert val_targets.shape == (
        len(val_loader.dataset),
        Config.num_classes,
    ), "Val target shape mismatch"
    assert 0.0 <= val_f1 <= 1.0, "F1 score must be between 0 and 1"

    # Save validation artifacts for inspection
    np.save(os.path.join(Config.working_dir, "val_logits.npy"), val_probs)
    np.save(os.path.join(Config.working_dir, "val_targets.npy"), val_targets)

    # 7. Threshold Optimization
    print("\n[7] Optimizing Threshold...")
    best_thresh, best_score = optimize_threshold(val_probs, val_targets)

    print(f"    Optimal Threshold: {best_thresh:.2f}")
    print(f"    Optimized Val F1: {best_score:.6f}")

    assert 0.0 < best_thresh < 1.0, "Threshold should be within (0, 1)"
    # Save best thresholds
    np.save(
        os.path.join(Config.working_dir, "best_thresholds.npy"), np.array([best_thresh])
    )

    # 8. Inference
    print("\n[8] Running Inference on Test Set (with TTA)...")
    # Using TTA as per Config
    test_probs, test_ids_out = inference(model, test_loader, device, use_tta=True)

    print(f"    Test Probs Shape: {test_probs.shape}")
    print(f"    Number of Test IDs: {len(test_ids_out)}")

    assert test_probs.shape == (
        len(test_loader.dataset),
        Config.num_classes,
    ), "Test probability shape mismatch"
    assert len(test_ids_out) == len(test_loader.dataset), "Test IDs count mismatch"

    # 9. Submission Generation
    print("\n[9] Generating Submission File...")

    # Apply optimized threshold
    test_preds_bin = (test_probs >= best_thresh).astype(int)

    submission_rows = []
    for i, img_id in enumerate(test_ids_out):
        # Get indices where prediction is 1
        pred_indices = np.where(test_preds_bin[i] == 1)[0]
        # Format as space-separated string
        pred_str = " ".join(map(str, pred_indices))
        submission_rows.append({"id": img_id, "attribute_ids": pred_str})

    df_sub = pd.DataFrame(submission_rows)

    # Save submission
    sub_path = os.path.join(Config.working_dir, "demo_submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"    Submission saved to: {sub_path}")

    # Verify submission format
    df_check = pd.read_csv(sub_path)
    print(f"    Submission Head:\n{df_check.head(2)}")

    assert list(df_check.columns) == [
        "id",
        "attribute_ids",
    ], "Submission columns mismatch"
    assert len(df_check) == len(test_loader.dataset), "Submission row count mismatch"

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(Config.working_dir, "demo_model.pth"))
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
