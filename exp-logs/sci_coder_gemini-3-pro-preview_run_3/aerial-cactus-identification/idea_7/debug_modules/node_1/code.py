import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_loaders, get_test_loader
from library.architectures import get_model
from library.engine import train_one_epoch, validate_one_epoch, predict_tta
from library.stacking import train_meta_learner, generate_submission


def main():
    print("Starting Cactus Identification Library Demo...")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config settings for speed and isolation
    Config.DEBUG = True  # Uses small subset (500 train, 100 test)
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_FOLDS = 2  # Minimum needed to verify CV logic conceptually

    # Use a specific directory for demo outputs to avoid cluttering main workspace
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up demo directory if it exists to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Loaders...")

    # Get Train/Val loaders for Fold 0
    # load_cached_data=False forces reprocessing of the debug subset
    train_loader, val_loader = get_loaders(fold_idx=0, load_cached_data=False)

    # Get Test loader
    test_loader, test_ids = get_test_loader(
        load_cached_data=True
    )  # Can use cache generated above

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), "Incorrect train image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect train label batch shape"
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Verify Test Loader
    test_images = next(iter(test_loader))
    print(f"Test Batch - Images: {test_images.shape}")
    assert test_images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), "Incorrect test image batch shape"
    assert len(test_ids) > 0, "Test IDs not loaded"

    print("Data Loader verification successful.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architectures...")

    dummy_input = torch.randn(2, 3, 32, 32).to(device)

    for model_name in Config.MODELS:
        print(f"Instantiating {model_name}...")
        model = get_model(model_name, num_classes=1, pretrained=False).to(device)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"  Output Shape: {output.shape}")

        # Assertions
        assert output.shape == (
            2,
            1,
        ), f"Model {model_name} output shape mismatch. Expected (2, 1), got {output.shape}"
        assert not torch.isnan(output).any(), f"Model {model_name} produced NaN outputs"

    print("Architecture verification successful.")

    # --------------------------------------------------------------------------
    # 4. Training Engine Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Training Engine...")

    # Initialize a model for training
    model = get_model("custom_wide_se_resnet", num_classes=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch
    print("Running training step...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )

    # Validate one epoch
    print("Running validation step...")
    val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

    print(
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
    )

    # Assertions
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # Test Time Augmentation (TTA) Prediction
    print("Running TTA prediction on validation set...")
    # Using a subset of val_loader for speed if needed, but debug set is already small
    preds = predict_tta(model, val_loader, device)

    assert len(preds) == len(val_loader.dataset), "TTA predictions count mismatch"
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions should be probabilities (0-1)"

    print("Training engine verification successful.")

    # --------------------------------------------------------------------------
    # 5. Stacking Logic Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Stacking Logic...")

    # Simulate OOF predictions and Targets for Stacking
    # We create synthetic data to test the stacking pipeline independently of model convergence
    num_samples = 100
    synthetic_targets = np.random.randint(0, 2, num_samples)

    # Create synthetic predictions for 3 models
    # Add some noise to targets to create realistic probabilities
    synthetic_oof_preds = {
        "model_A": np.clip(
            synthetic_targets * 0.8 + np.random.rand(num_samples) * 0.2, 0, 1
        ),
        "model_B": np.clip(
            synthetic_targets * 0.7 + np.random.rand(num_samples) * 0.3, 0, 1
        ),
        "model_C": np.clip(
            synthetic_targets * 0.9 + np.random.rand(num_samples) * 0.1, 0, 1
        ),
    }

    # Train Meta Learner
    print("Training Meta Learner...")
    meta_learner, oof_auc = train_meta_learner(synthetic_oof_preds, synthetic_targets)

    assert (
        oof_auc > 0.5
    ), "Meta learner AUC should be reasonable on synthetic correlated data"

    # Simulate Test predictions
    num_test_samples = 50
    synthetic_test_ids = [f"img_{i}.jpg" for i in range(num_test_samples)]
    synthetic_test_preds = {
        "model_A": np.random.rand(num_test_samples),
        "model_B": np.random.rand(num_test_samples),
        "model_C": np.random.rand(num_test_samples),
    }

    # Generate Submission
    print("Generating Submission...")
    sub_df = generate_submission(
        meta_learner,
        synthetic_test_preds,
        synthetic_test_ids,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission
    print("Verifying Submission File...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    loaded_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert loaded_df.shape == (num_test_samples, 2), "Submission shape mismatch"
    assert list(loaded_df.columns) == [
        "id",
        "has_cactus",
    ], "Submission columns mismatch"
    assert (
        loaded_df["has_cactus"].between(0, 1).all()
    ), "Submission probabilities out of range"

    print("Stacking logic verification successful.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
