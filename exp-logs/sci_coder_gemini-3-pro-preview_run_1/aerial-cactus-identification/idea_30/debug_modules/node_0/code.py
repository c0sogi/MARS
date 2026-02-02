import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, cache_dataset_in_ram
from library.models import CactusRepVGG
from library.engine import train_one_epoch, validate, SWAHandler
from library.stacking import run_stacking, StackingDataManager


def run_demo():
    print(">>> Starting Cactus Classification Pipeline Demo")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We override the Config class attributes to create an isolated demo environment
    # and reduce runtime (fewer epochs, smaller batches).
    print(">>> Configuring environment...")

    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # We must manually update the dependent path variables since they were initialized
    # when the class was defined.
    Config.CACHE_TRAIN_IMGS = os.path.join(Config.CACHE_DIR, "train_imgs.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(Config.CACHE_DIR, "train_labels.npy")
    Config.CACHE_TRAIN_FSIZES = os.path.join(Config.CACHE_DIR, "train_fsizes.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(Config.CACHE_DIR, "train_ids.npy")
    Config.CACHE_TEST_IMGS = os.path.join(Config.CACHE_DIR, "test_imgs.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.CACHE_DIR, "test_ids.npy")
    Config.CACHE_TEST_FSIZES = os.path.join(Config.CACHE_DIR, "test_fsizes.npy")

    # Hyperparameters for fast execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 2  # Reduced workers for demo
    Config.SWA_START_EPOCH = 0  # Allow SWA to run immediately for demo

    # Create directories and set seed
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> [1/5] Testing Data Loading...")

    # load_cached_data=False forces the processing of raw images from ./input
    # This ensures the data pipeline is fully exercised.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Verify batch structure
    batch = next(iter(train_loader))
    imgs, lbls, fsizes, ids = batch

    print(
        f"   Batch Shapes -> Images: {imgs.shape}, Labels: {lbls.shape}, FileSizes: {fsizes.shape}"
    )

    # Assertions
    assert imgs.shape == (Config.BATCH_SIZE, 3, 32, 32), "Image batch shape incorrect"
    assert lbls.shape == (Config.BATCH_SIZE,), "Label batch shape incorrect"
    assert fsizes.shape == (Config.BATCH_SIZE,), "FileSize batch shape incorrect"
    assert isinstance(ids, tuple) or isinstance(ids, list), "IDs should be a list/tuple"
    assert len(ids) == Config.BATCH_SIZE, "ID list length incorrect"

    # -------------------------------------------------------------------------
    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n>>> [2/5] Testing Model Instantiation (RepVGG)...")

    model = CactusRepVGG(num_classes=1).to(Config.DEVICE)

    # Dummy forward pass
    dummy_input = torch.randn(4, 3, 32, 32).to(Config.DEVICE)
    cls_logits, quality_pred = model(dummy_input)

    print(
        f"   Output Shapes -> Logits: {cls_logits.shape}, Quality: {quality_pred.shape}"
    )

    assert cls_logits.shape == (4, 1), "Class logits shape incorrect"
    assert quality_pred.shape == (4, 1), "Quality prediction shape incorrect"

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n>>> [3/5] Testing Training Engine (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Train one epoch
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, Config.DEVICE, epoch=0
    )
    print(f"   Train Metrics: {train_metrics}")

    # Validate
    val_metrics = validate(model, val_loader, Config.DEVICE)
    print(f"   Val Metrics: {val_metrics}")

    # Check if metrics are populated
    assert train_metrics.metrics["Loss"]["count"] > 0, "Training did not update metrics"
    assert "AUC" in val_metrics.metrics, "Validation missing AUC metric"

    # -------------------------------------------------------------------------
    # 5. SWA (Stochastic Weight Averaging)
    # -------------------------------------------------------------------------
    print("\n>>> [4/5] Testing SWA Handler...")

    swa_handler = SWAHandler(model)

    # Simulate an SWA update step
    swa_handler.step(model, epoch=Config.SWA_START_EPOCH + 1)

    # Finalize SWA (Update BN statistics)
    # This iterates through the train_loader to update running stats
    print("   Finalizing SWA (updating BN stats)...")
    swa_handler.finalize(model, train_loader, Config.DEVICE)

    print("   SWA Finalization successful.")

    # -------------------------------------------------------------------------
    # 6. Stacking Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> [5/5] Testing Stacking Pipeline...")

    # Initialize Data Manager with our demo working directory
    dm = StackingDataManager(working_dir=Config.WORKING_DIR)

    # Load ground truth (this uses the cache we just created in step 2)
    gt_data = dm.load_ground_truth()

    n_train = len(gt_data["train_labels"])
    n_test = len(gt_data["test_ids"])

    print(f"   Ground Truth Loaded: Train={n_train}, Test={n_test}")

    # Generate Mock Predictions for Stacking
    # We simulate 2 models: 'Model_A' and 'Model_B'
    np.random.seed(Config.SEED)

    oof_predictions = {
        "Model_A": {
            "probs": np.random.rand(n_train),
            "fsizes": np.random.rand(n_train),
        },
        "Model_B": {
            "probs": np.random.rand(n_train),
            "fsizes": np.random.rand(n_train),
        },
    }

    test_predictions = {
        "Model_A": {"probs": np.random.rand(n_test), "fsizes": np.random.rand(n_test)},
        "Model_B": {"probs": np.random.rand(n_test), "fsizes": np.random.rand(n_test)},
    }

    # Run Stacking
    # load_cache=False ensures we compute features from scratch
    final_probs = run_stacking(oof_predictions, test_predictions, load_cache=False)

    # Verification
    assert len(final_probs) == n_test, "Final predictions length mismatch"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify Submission File Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission File Created: {Config.SUBMISSION_PATH}")
    print(f"   Submission Shape: {df_sub.shape}")

    assert df_sub.shape == (n_test, 2), "Submission CSV shape incorrect"
    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns incorrect"

    print("\n>>> DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_demo()
