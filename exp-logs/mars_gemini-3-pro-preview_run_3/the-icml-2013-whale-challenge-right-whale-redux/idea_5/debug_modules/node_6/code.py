import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_auc, calculate_pos_weight
from library.dataset import get_dataloaders, AudioPreprocessor
from library.model import WhaleModel
from library.loss import WeightedBCELoss, MixupLoss
from library.train import Trainer


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration Override for Demo/Speed
    # --------------------------------------------------------------------------
    print(">>> Configuring environment for demonstration...")

    # Override Config attributes to run a fast, small-scale test
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Process only 20 files for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug run

    # Redirect outputs to a demo directory
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.MODEL_CHECKPOINT_DIR = Config.WORK_DIR
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration complete.")

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Utility Functions...")

    # Test AUC Calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_auc(y_true, y_pred)
    print(f"Calculated AUC (dummy data): {auc}")
    assert 0 <= auc <= 1, "AUC must be between 0 and 1"

    # Test Positive Weight Calculation
    if os.path.exists(Config.TRAIN_CSV):
        pos_weight = calculate_pos_weight(Config.TRAIN_CSV)
        print(f"Calculated Positive Weight from Metadata: {pos_weight.item():.4f}")
        assert isinstance(pos_weight, torch.Tensor), "Pos weight must be a tensor"
    else:
        print(f"Warning: {Config.TRAIN_CSV} not found. Skipping weight check.")

    # --------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Dataset and DataLoaders...")

    # Generate DataLoaders (this triggers caching/processing)
    # load_cached_data=False forces reprocessing to demonstrate the pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch
    inputs, targets, clips = next(iter(train_loader))

    print(f"Batch Input Shape: {inputs.shape}")  # Expected: (B, 3, 320, 200)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (B,)

    # Assertions
    assert inputs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert inputs.shape[1] == Config.IN_CHANNELS, "Channel count mismatch"
    assert inputs.shape[2] == Config.IMG_SIZE[0], "Height mismatch"
    assert inputs.shape[3] == Config.IMG_SIZE[1], "Width mismatch"
    assert len(clips) == Config.BATCH_SIZE, "Clip list length mismatch"

    # --------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    # Initialize Model
    # We use pretrained=False here just to avoid downloading weights during this quick check,
    # though the real run uses True.
    model = WhaleModel(pretrained=False)
    model.to(Config.DEVICE)

    # Forward Pass
    inputs = inputs.to(Config.DEVICE)
    outputs = model(inputs)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    # --------------------------------------------------------------------------
    # 5. Verify Loss Functions
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Loss Functions...")

    # Weighted BCE
    criterion = WeightedBCELoss(pos_weight=torch.tensor([1.0]), device=Config.DEVICE)
    targets = targets.to(Config.DEVICE)

    loss = criterion(outputs, targets)
    print(f"Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # Mixup Loss
    mixup_criterion = MixupLoss(criterion)
    lam = 0.5
    # Simulate shuffled targets
    targets_b = targets.flip(0)
    mixup_loss = mixup_criterion(outputs, targets, targets_b, lam)
    print(f"Calculated Mixup Loss: {mixup_loss.item():.6f}")
    assert not torch.isnan(mixup_loss), "Mixup Loss is NaN"

    # --------------------------------------------------------------------------
    # 6. Verify Training Loop (Trainer)
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Trainer (Fit & Predict)...")

    trainer = Trainer(debug=True)

    # Run Training (1 Epoch on subset)
    trainer.fit()

    # Run Prediction
    trainer.predict()

    # Check if submission file was created
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file successfully created at: {Config.SUBMISSION_PATH}")

        # Validate content format
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission Head:")
        print(df_sub.head())

        assert "clip" in df_sub.columns, "Submission missing 'clip' column"
        assert (
            "probability" in df_sub.columns
        ), "Submission missing 'probability' column"
        assert len(df_sub) > 0, "Submission file is empty"
        assert (
            df_sub["probability"].dtype == float
            or df_sub["probability"].dtype == np.float32
            or df_sub["probability"].dtype == np.float64
        ), "Probability column is not float"

    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n>>> All demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
