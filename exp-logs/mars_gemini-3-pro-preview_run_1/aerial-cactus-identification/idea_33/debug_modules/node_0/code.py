import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.models import CactusRepVGG, CactusResNet
from library.engine import Engine
from library.stacking import StackingEnsemble


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    print("--- Setting up Configuration ---")

    # Override Config for a fast, isolated demo run
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Enable Debug mode to use only 100 samples per dataset
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2  # Reduced workers for small data

    # Clean up any previous demo artifacts to ensure a fresh run
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)

    # Create directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    Config.print_config()

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n--- Loading Data ---")
    # get_dataloaders handles caching and loading. With DEBUG=True, it loads a subset.
    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Data Loading
    train_batch, train_targets = next(iter(train_loader))
    print(f"Train Batch Shape: {train_batch.shape}")
    print(f"Train Targets Shape: {train_targets.shape}")

    assert train_batch.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), "Incorrect train image shape"
    assert train_targets.shape == (Config.BATCH_SIZE,), "Incorrect train target shape"
    assert train_batch.dtype == torch.float32, "Images should be float32"
    assert (
        train_batch.max() <= 1.0 and train_batch.min() >= 0.0
    ), "Images should be normalized [0,1]"

    # --------------------------------------------------------------------------
    # 3. Model Training (Base Learners)
    # --------------------------------------------------------------------------
    print("\n--- Training Base Learners ---")
    device = Config.DEVICE

    # Define models to train. Using smaller ResNet layers for speed.
    models_to_train = [
        ("CactusRepVGG", CactusRepVGG(num_classes=1)),
        ("CactusResNet", CactusResNet(num_classes=1, layers=[1, 1, 1, 1])),
    ]

    trained_model_paths = []

    for name, model in models_to_train:
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # Train for one epoch
        loss = Engine.train_one_epoch(model, train_loader, optimizer, device, epoch=1)

        # Verify loss is valid
        assert not np.isnan(loss), f"Training loss is NaN for {name}"

        # Validate
        val_loss, val_auc = Engine.validate(model, val_loader, device)
        print(f"{name} - Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Save Checkpoint
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{name}_best.pth")
        torch.save(model.state_dict(), ckpt_path)
        trained_model_paths.append(ckpt_path)

        # Cleanup to free memory
        del model, optimizer
        torch.cuda.empty_cache()

    assert len(trained_model_paths) == 2, "Should have trained 2 models"

    # --------------------------------------------------------------------------
    # 4. Stacking Ensemble
    # --------------------------------------------------------------------------
    print("\n--- Stacking Ensemble ---")
    stacker = StackingEnsemble()

    # Prepare validation targets for the meta-learner
    # Since we are in DEBUG mode, iterate the loader to get the subset targets
    val_targets_all = []
    for _, t in val_loader:
        val_targets_all.extend(t.numpy())
    val_targets_all = np.array(val_targets_all)

    # Fit the meta-learner
    # This extracts geometric features (mean, std) from TTA predictions of base models
    # and trains a LogisticRegression on top.
    print("Fitting Meta-Learner...")
    stacking_auc = stacker.fit(trained_model_paths, val_loader, val_targets_all, device)

    print(f"Stacking Validation AUC: {stacking_auc:.4f}")
    assert 0.0 <= stacking_auc <= 1.0, "AUC must be between 0 and 1"

    # --------------------------------------------------------------------------
    # 5. Inference and Submission
    # --------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    # Extract Test IDs (needed for submission file)
    test_ids_all = []
    for _, ids in test_loader:
        test_ids_all.extend(ids)
    test_ids_all = np.array(test_ids_all)

    # Create submission file
    stacker.create_submission(
        model_paths=trained_model_paths,
        test_loader=test_loader,
        test_ids=test_ids_all,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("\nSubmission File Preview:")
    print(df_sub.head())

    # Validation checks on submission
    assert len(df_sub) == len(
        test_ids_all
    ), f"Submission row count mismatch. Expected {len(test_ids_all)}, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "id",
        "has_cactus",
    ], "Incorrect columns in submission"
    assert df_sub["has_cactus"].dtype == float, "Prediction column should be float"
    assert (
        df_sub["has_cactus"].min() >= 0.0 and df_sub["has_cactus"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
