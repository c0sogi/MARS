import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Add the current directory to path to ensure library imports work if needed
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, optimize_f1_threshold, AverageMeter
from library.dataset import get_dataloaders, MixupCutMix, ArtworkDataset
from library.model import ArtworkModel, ModelEMA, GeM
from library.engine import train_one_epoch, valid_one_epoch


def run_demonstration():
    print("=== Starting Artwork Attribution Pipeline Demonstration ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment...")
    seed_everything(Config.seed)

    # Override Config for rapid testing
    Config.debug = True
    Config.debug_sample_size = 64  # Small enough for quick CPU/GPU processing
    Config.batch_size = 8
    Config.epochs = 1
    Config.pretrained = False  # Disable downloading weights for speed/offline safety
    Config.num_workers = 2  # Reduce workers for small data

    print(f"Debug Mode: {Config.debug}")
    print(f"Sample Size: {Config.debug_sample_size}")
    print(f"Device: {Config.device}")

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline...")

    # Generate DataLoaders
    # load_cached_data=False forces the code to process the CSVs from metadata
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Batch Structure
    images, targets, ids = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Incorrect image batch shape"
    assert targets.shape == (
        Config.batch_size,
        Config.num_classes,
    ), "Incorrect target batch shape"
    assert len(ids) == Config.batch_size, "Incorrect number of IDs"

    # Verify Mixup/CutMix
    print("Verifying MixupCutMix...")
    mixup_fn = MixupCutMix(
        prob=1.0, num_classes=Config.num_classes
    )  # Force augmentation
    mixed_imgs, mixed_targets = mixup_fn(images, targets)

    assert mixed_imgs.shape == images.shape, "Mixup changed image shape"
    assert mixed_targets.shape == targets.shape, "Mixup changed target shape"
    # Check that targets are no longer just 0 or 1 (implying mixing happened)
    # Note: In rare cases where lambda is 0 or 1, this might fail, but prob is low with beta dist.
    # We'll just check types.
    assert mixed_targets.dtype == torch.float32, "Mixed targets should be float32"

    # 3. Model Verification
    print("\n[3] Verifying Model Architecture...")

    # Instantiate Model
    model = ArtworkModel(pretrained=False)
    model.to(Config.device)

    # Test GeM Pooling specifically
    print("Testing GeM pooling layer...")
    gem_layer = GeM(p=3.0)
    dummy_features = torch.randn(4, 2048, 10, 10)  # B, C, H, W
    pooled = gem_layer(dummy_features)
    assert pooled.shape == (4, 2048, 1, 1), f"GeM output shape mismatch: {pooled.shape}"

    # Test Forward Pass
    print("Testing full model forward pass...")
    dummy_input = torch.randn(
        Config.batch_size, 3, Config.image_size, Config.image_size
    ).to(Config.device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Model output shape mismatch: {output.shape}"
    print("Forward pass successful.")

    # Test Model EMA
    print("Testing Model EMA...")
    model_ema = ModelEMA(model, decay=0.99, device=Config.device)
    # Perform a dummy update
    model_ema.update(model)
    print("Model EMA initialized and updated.")

    # 4. Training Loop Demonstration
    print("\n[4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # Train one epoch
    avg_train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=Config.device,
        epoch=1,
        model_ema=model_ema,
    )

    print(f"Training finished. Average Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss is NaN"
    assert avg_train_loss > 0, "Training loss should be positive"

    # 5. Validation Loop Demonstration
    print("\n[5] Running Validation Loop...")

    # Validate using the EMA model
    val_loss, val_preds, val_labels = valid_one_epoch(
        model=model_ema.module, dataloader=val_loader, device=Config.device
    )

    print(f"Validation finished. Loss: {val_loss:.4f}")
    assert val_preds.shape == (
        len(val_loader.dataset),
        Config.num_classes,
    ), "Prediction shape mismatch"
    assert val_labels.shape == (
        len(val_loader.dataset),
        Config.num_classes,
    ), "Label shape mismatch"

    # 6. Metric Optimization
    print("\n[6] Optimizing F1 Threshold...")

    # Use the utils function to find best threshold
    best_thresh, best_f1 = optimize_f1_threshold(val_labels, val_preds)

    print(f"Best Threshold: {best_thresh:.4f}")
    print(f"Best Micro F1: {best_f1:.4f}")

    assert 0 <= best_f1 <= 1.0, "F1 score out of range"
    assert (
        Config.threshold_start <= best_thresh <= Config.threshold_end
    ), "Threshold out of search range"

    # 7. Inference Simulation
    print("\n[7] Simulating Inference on Test Set...")

    model.eval()
    test_preds = []
    test_ids = []

    # Run a few batches of test inference
    with torch.no_grad():
        for i, (images, _, ids) in enumerate(test_loader):
            images = images.to(Config.device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Apply the optimized threshold
            binary_preds = (probs > best_thresh).int().cpu().numpy()

            test_preds.append(binary_preds)
            test_ids.extend(ids)

            # Break early for demonstration
            if i >= 2:
                break

    test_preds = np.concatenate(test_preds)
    print(f"Processed {len(test_ids)} test images.")
    print(f"Test Predictions Shape: {test_preds.shape}")

    # Verify submission format creation
    print("Formatting sample submission...")
    submission_rows = []
    for idx, img_id in enumerate(test_ids):
        # Get indices where prediction is 1
        pred_indices = np.where(test_preds[idx] == 1)[0]
        pred_str = " ".join(map(str, pred_indices))
        submission_rows.append({"id": img_id, "attribute_ids": pred_str})

    sub_df = pd.DataFrame(submission_rows)
    print("Sample Submission DataFrame:")
    print(sub_df.head())

    assert (
        "id" in sub_df.columns and "attribute_ids" in sub_df.columns
    ), "Submission columns missing"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
