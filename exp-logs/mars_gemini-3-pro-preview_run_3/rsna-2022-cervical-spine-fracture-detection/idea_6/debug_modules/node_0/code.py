import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_weighted_log_loss
from library.data import cache_dataset, CervicalSpineDataset, VolumetricTransforms
from library.model import AnatomicallyConditionedResNet
from library.loss import HierarchicalCompoundLoss
from library.engine import get_optimizer_and_scheduler, train_one_epoch, evaluate


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Configuration
    # Enable debug mode to reduce epochs (2) and batch size (4)
    Config.setup(debug=True)
    seed_everything(Config.SEED)

    # Define a temporary cache directory for this demo to avoid conflicts
    demo_cache_dir = os.path.join(Config.WORKING_DIR, "demo_cache")
    os.makedirs(demo_cache_dir, exist_ok=True)

    # 2. Data Preparation
    print("\n[Data] Loading metadata...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select a small subset for demonstration (4 samples to fit in one batch of size 4)
    subset_df = full_train_df.head(4).copy()
    print(f"[Data] Selected subset of {len(subset_df)} samples for demonstration.")

    # Run Preprocessing/Caching
    # This demonstrates loading DICOMs, windowing, resizing, and saving .npy volumes
    print("[Data] Caching dataset subset...")
    cache_dataset(subset_df, demo_cache_dir, load_cached_data=False)

    # Verify cache files exist
    for uid in subset_df["StudyInstanceUID"]:
        expected_path = os.path.join(demo_cache_dir, f"{uid}.npy")
        if not os.path.exists(expected_path):
            raise FileNotFoundError(f"Cache file failed to generate: {expected_path}")

    # Instantiate Dataset
    # We use a simple transform pipeline or None for demo
    transforms = VolumetricTransforms(prob=0.5)
    train_dataset = CervicalSpineDataset(
        subset_df, demo_cache_dir, mode="train", transforms=transforms
    )

    # Instantiate DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # False for reproducibility in demo
        num_workers=0,  # 0 for simple main-thread execution in demo
        pin_memory=True,
    )

    # Verify Data Loading
    print("[Data] Verifying DataLoader output...")
    sample_imgs, sample_labels = next(iter(train_loader))

    # Expected Input Shape: (Batch, Slices, Channels, H, W)
    # Config.NUM_SLICES=64, Channels=3 (2.5D), H=W=224
    expected_shape = (Config.BATCH_SIZE, Config.NUM_SLICES, 3, 224, 224)
    assert (
        sample_imgs.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {sample_imgs.shape}"

    # Expected Label Shape: (Batch, 8) -> [C1..C7, patient_overall]
    assert sample_labels.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE, 8)}, got {sample_labels.shape}"
    print("    -> Data shapes correct.")

    # 3. Model Instantiation & Verification
    print("\n[Model] Instantiating AnatomicallyConditionedResNet...")
    model = AnatomicallyConditionedResNet().to(Config.DEVICE)

    # Forward Pass Check
    print("[Model] Running forward pass on sample batch...")
    sample_imgs = sample_imgs.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(sample_imgs)

    # Expected Output: (Batch, 7) -> Logits for C1-C7
    assert logits.shape == (
        Config.BATCH_SIZE,
        7,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 7)}, got {logits.shape}"
    print("    -> Forward pass successful.")

    # 4. Loss Function Verification
    print("\n[Loss] Verifying HierarchicalCompoundLoss...")
    loss_fn = HierarchicalCompoundLoss()

    # Create dummy data for logic check
    # Batch of 2
    # Case 1: All logits very low (predicting no fracture), Target no fracture
    # Case 2: C1 logit high (predicting fracture), Target C1 fracture + Patient fracture
    dummy_logits = torch.tensor(
        [
            [-10.0] * 7,  # Case 1
            [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # Case 2
        ]
    ).to(Config.DEVICE)

    dummy_targets = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]]  # Case 1  # Case 2
    ).to(Config.DEVICE)

    loss_val = loss_fn(dummy_logits, dummy_targets)
    assert loss_val.ndim == 0, "Loss should be a scalar."
    assert loss_val > 0, "Loss should be positive."
    print(f"    -> Loss calculation successful. Value: {loss_val.item():.4f}")

    # 5. Metric Verification
    print("\n[Metric] Verifying Weighted Log Loss...")
    # Perfect predictions
    y_true_np = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]])
    y_pred_perfect = np.array(
        [[1e-5] * 7 + [1e-5], [0.999, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 0.999]]
    )

    metric_perfect = get_weighted_log_loss(y_pred_perfect, y_true_np)
    print(f"    -> Perfect prediction loss: {metric_perfect:.6f}")
    assert metric_perfect < 0.01, "Perfect predictions should have near-zero loss."

    # 6. Training Loop Demonstration
    print("\n[Engine] Starting Training Loop Demonstration...")
    optimizer, scheduler = get_optimizer_and_scheduler(model, Config.EPOCHS)

    # Train one epoch
    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, Config.DEVICE, epoch=0
    )
    print(f"    -> Epoch 0 Train Loss: {avg_train_loss:.4f}")

    # Evaluate
    # Using the same loader as validation for demo purposes
    avg_val_loss, val_metric = evaluate(model, train_loader, Config.DEVICE)
    print(f"    -> Validation Loss: {avg_val_loss:.4f}")
    print(f"    -> Validation Metric (Weighted Log Loss): {val_metric:.4f}")

    # 7. Inference / Submission Structure
    print("\n[Inference] Simulating Test Prediction...")
    # In test mode, dataset returns (image, uid)
    test_dataset = CervicalSpineDataset(subset_df, demo_cache_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    model.eval()
    predictions = []

    with torch.no_grad():
        for imgs, uids in test_loader:
            imgs = imgs.to(Config.DEVICE)
            logits = model(imgs)  # (B, 7)
            probs = torch.sigmoid(logits)

            # Derive patient_overall
            patient_prob = torch.max(probs, dim=1).values.unsqueeze(1)

            # Combine
            batch_preds = torch.cat([probs, patient_prob], dim=1).cpu().numpy()

            for i, uid in enumerate(uids):
                row_preds = batch_preds[i]  # 8 values
                predictions.append((uid, row_preds))

    print(f"    -> Generated predictions for {len(predictions)} studies.")

    # Verify submission format logic
    # We need 8 rows per study
    submission_rows = []
    col_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    for uid, preds in predictions:
        for val, col in zip(preds, col_names):
            row_id = f"{uid}_{col}"
            submission_rows.append({"row_id": row_id, "fractured": val})

    submission_df = pd.DataFrame(submission_rows)
    print("    -> Sample Submission DataFrame:")
    print(submission_df.head(8).to_string(index=False))

    assert len(submission_df) == len(subset_df) * 8, "Submission row count mismatch."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
