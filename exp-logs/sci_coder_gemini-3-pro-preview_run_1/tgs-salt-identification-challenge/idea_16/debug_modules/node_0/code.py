import os
import shutil
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_decode, rle_encode
from library.dataset import get_loaders
from library.model import DeepResUNet
from library.losses import CurriculumLoss
from library.train import Trainer
from library.inference import predict
from library.metrics import calculate_map_at_thresholds

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Salt Segmentation Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Demo Configuration...")

    # Define a separate working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "demo_submission")
    Config.CACHE_DIR = DEMO_DIR  # Cache processed numpy arrays here

    # Override Data paths (will point to mini-CSVs created in step 2)
    METADATA_DEMO_DIR = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(METADATA_DEMO_DIR, exist_ok=True)

    Config.TRAIN_CSV = os.path.join(METADATA_DEMO_DIR, "train.csv")
    Config.VAL_CSV = os.path.join(METADATA_DEMO_DIR, "val.csv")
    Config.TEST_CSV = os.path.join(METADATA_DEMO_DIR, "test.csv")

    # Override Training Hyperparameters for Speed
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.TOTAL_EPOCHS = 2
    Config.EPOCHS_PER_CYCLE = 1
    Config.CYCLE_1_END_EPOCH = 1
    Config.SAVE_CYCLES = [1, 2]  # Save checkpoints at epoch 1 and 2

    # Setup directories based on new config
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Prepare Mini-Dataset
    # -------------------------------------------------------------------------
    print("\n[2] Preparing Mini-Dataset (Subsetting)...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample a small subset (enough for a few batches)
    # Batch size is 4, so we take 8 samples for train, 4 for val, 4 for test
    mini_train = orig_train.head(8).copy()
    mini_val = orig_val.head(4).copy()
    mini_test = orig_test.head(4).copy()

    # Save mini metadata
    mini_train.to_csv(Config.TRAIN_CSV, index=False)
    mini_val.to_csv(Config.VAL_CSV, index=False)
    mini_test.to_csv(Config.TEST_CSV, index=False)

    print(f"    Created mini_train.csv with {len(mini_train)} rows.")
    print(f"    Created mini_val.csv with {len(mini_val)} rows.")
    print(f"    Created mini_test.csv with {len(mini_test)} rows.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Loading...")

    # This will process the mini-CSVs and cache .npy files in Config.CACHE_DIR
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Verify Train Loader
    images, masks, ids = next(iter(train_loader))
    print(f"    Train Batch Images Shape: {images.shape}")
    print(f"    Train Batch Masks Shape: {masks.shape}")

    # Assertions
    # Expected shape: (Batch, Channels, Height, Width)
    # Config.INPUT_CHANNELS = 2 (Image + Depth)
    # Config.IMG_HEIGHT/WIDTH = 128
    assert images.shape == (4, 2, 128, 128), "Incorrect train image batch shape"
    assert masks.shape == (4, 1, 128, 128), "Incorrect train mask batch shape"
    assert len(ids) == 4, "Incorrect number of IDs"

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Initialization and Forward Pass...")

    model = DeepResUNet().to(device)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)

    # Forward pass
    logits = model(images)
    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (4, 1, 128, 128), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN values"

    # -------------------------------------------------------------------------
    # 5. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[5] Testing Loss Function (CurriculumLoss)...")

    criterion = CurriculumLoss()

    # Test Phase 1 (BCE + Dice)
    loss_phase_1 = criterion(logits, masks, epoch=0)
    print(f"    Phase 1 Loss (Epoch 0): {loss_phase_1.item():.4f}")

    # Test Phase 2 (BCE + Lovasz) - simulating epoch >= CYCLE_1_END_EPOCH
    loss_phase_2 = criterion(logits, masks, epoch=Config.CYCLE_1_END_EPOCH + 1)
    print(
        f"    Phase 2 Loss (Epoch {Config.CYCLE_1_END_EPOCH + 1}): {loss_phase_2.item():.4f}"
    )

    assert loss_phase_1 > 0, "Loss should be positive"
    assert loss_phase_2 > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (2 Epochs)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=1e-6
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
    )

    # Run training
    trainer.fit(num_epochs=Config.TOTAL_EPOCHS)

    # Verify Checkpoints
    expected_checkpoints = ["best_model.pth", "best_cycle_2.pth"]
    # Note: cycle logic in Trainer: epoch 0 -> cycle 1, epoch 1 -> cycle 2 (since EPOCHS_PER_CYCLE=1)
    # Trainer saves cycle 2 and 3. Here we run 2 epochs.
    # Epoch 0 (Cycle 1): Saves best_model.pth if best.
    # Epoch 1 (Cycle 2): Saves best_cycle_2.pth if best.

    print("    Verifying checkpoints...")
    for ckpt in expected_checkpoints:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt)
        if os.path.exists(ckpt_path):
            print(f"    [OK] Found checkpoint: {ckpt}")
        else:
            # It's possible validation score didn't improve, but with random init and 1 batch it usually does.
            # We won't crash here, but we'll note it.
            print(
                f"    [INFO] Checkpoint {ckpt} not found (metric might not have improved)."
            )

    # Ensure at least one model exists for inference
    if not os.listdir(Config.CHECKPOINT_DIR):
        print(
            "    [WARNING] No checkpoints saved. Saving manual checkpoint for inference demo."
        )
        torch.save(
            model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        )

    # -------------------------------------------------------------------------
    # 7. Inference
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference...")

    # To ensure inference works even if cycle checkpoints weren't saved due to metric logic,
    # we mock the cycle checkpoints if they are missing by copying best_model or current model.
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    cycle2_path = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_2.pth")
    cycle3_path = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_3.pth")

    if os.path.exists(best_model_path) and not os.path.exists(cycle2_path):
        shutil.copy(best_model_path, cycle2_path)
    if os.path.exists(best_model_path) and not os.path.exists(cycle3_path):
        shutil.copy(best_model_path, cycle3_path)

    # Run inference
    predict(limit_batches=None)

    # Verify Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created"

    df_sub = pd.read_csv(sub_path)
    print(f"    Submission file loaded. Rows: {len(df_sub)}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    assert len(df_sub) == 4, "Submission should have 4 rows (matching mini_test)"
    assert "rle_mask" in df_sub.columns, "Missing rle_mask column"

    # -------------------------------------------------------------------------
    # 8. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[8] Verifying Metric Logic (mAP @ IoU)...")

    # Case 1: Perfect Match
    t_pred = torch.ones((1, 1, 10, 10))
    t_target = torch.ones((1, 1, 10, 10))
    score_perfect = calculate_map_at_thresholds(t_pred, t_target, threshold=0.5)
    print(f"    Perfect Match Score: {score_perfect}")
    assert score_perfect == 1.0, "Perfect match should be 1.0"

    # Case 2: No Overlap
    t_pred_zero = torch.zeros((1, 1, 10, 10))
    score_mismatch = calculate_map_at_thresholds(t_pred_zero, t_target, threshold=0.5)
    print(f"    No Overlap Score: {score_mismatch}")
    assert score_mismatch == 0.0, "No overlap should be 0.0"

    # Case 3: Empty Target, Empty Prediction (True Negative)
    score_tn = calculate_map_at_thresholds(t_pred_zero, t_pred_zero, threshold=0.5)
    print(f"    Empty/Empty Score: {score_tn}")
    assert score_tn == 1.0, "Correctly predicting empty should be 1.0"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
