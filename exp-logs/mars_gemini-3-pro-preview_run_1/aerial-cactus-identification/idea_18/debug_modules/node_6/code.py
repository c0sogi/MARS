import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import CactusDataset, get_transforms
from library.models import CactusModel
from library.stacking import StackingTrainer


# =============================================================================
# 1. Configuration Setup for Demo/Fast Run
# =============================================================================
def setup_demo_config():
    """
    Overrides default Config parameters to run a fast demonstration.
    """
    print(">>> Setting up Demo Configuration...")

    # 1. Change Working Directory to a separate demo folder
    Config.WORKING_DIR = "./working/demo_run_script"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update Cache Paths to point to the new demo directory
    Config.CACHE_FILE_MAP = {
        "train_imgs": os.path.join(Config.CACHE_DIR, "cache_train_imgs.npy"),
        "train_labels": os.path.join(Config.CACHE_DIR, "cache_train_labels.npy"),
        "train_fsizes": os.path.join(Config.CACHE_DIR, "cache_train_filesizes.npy"),
        "train_ids": os.path.join(Config.CACHE_DIR, "cache_train_ids.npy"),
        "test_imgs": os.path.join(Config.CACHE_DIR, "cache_test_imgs.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "cache_test_ids.npy"),
        "test_fsizes": os.path.join(Config.CACHE_DIR, "cache_test_filesizes.npy"),
        "val_imgs": os.path.join(Config.CACHE_DIR, "cache_val_imgs.npy"),
        "val_labels": os.path.join(Config.CACHE_DIR, "cache_val_labels.npy"),
        "val_fsizes": os.path.join(Config.CACHE_DIR, "cache_val_filesizes.npy"),
    }

    # 2. Reduce Training Complexity
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_FOLDS = 2  # Use only 2 folds
    Config.BATCH_SIZE = 16  # Small batch size
    Config.DEBUG = True

    # 3. Use a Single Model for speed (RepVGG Spatial)
    Config.MODEL_CONFIGS = {
        "RepVGG_Spatial": {
            "arch": "RepVGG",
            "in_chans": 3,
            "use_film": True,
            "use_mtl": True,
        }
    }

    # Initialize directories
    Config.setup_directories()

    # Set Seed
    seed_everything(Config.SEED)
    print(">>> Configuration Updated for Demo.\n")


# =============================================================================
# 2. Unit Testing: Dataset and Model
# =============================================================================
def test_dataset_and_model():
    """
    Verifies that the Dataset produces correct shapes and the Model performs a forward pass.
    """
    print(">>> Starting Unit Test: Dataset & Model...")

    # --- Mock Data ---
    N = 32
    H, W = 32, 32
    mock_images = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8)
    mock_labels = np.random.randint(0, 2, (N,)).astype(np.float32)
    mock_fsizes = np.random.randint(500, 5000, (N,)).astype(np.float32)
    mock_ids = np.array([f"img_{i}.jpg" for i in range(N)])

    # --- Test Dataset ---
    dataset = CactusDataset(
        images=mock_images,
        labels=mock_labels,
        file_sizes=mock_fsizes,
        ids=mock_ids,
        transform=get_transforms("train", in_chans=3),
        in_chans=3,
    )

    # Get one sample
    sample = dataset[0]

    # Assertions for Dataset
    assert "image" in sample
    assert "label" in sample
    assert "file_size_norm" in sample
    assert "file_size_log" in sample
    assert sample["image"].shape == (
        3,
        32,
        32,
    ), f"Expected (3, 32, 32), got {sample['image'].shape}"
    assert isinstance(sample["label"], torch.Tensor)
    print("    [Pass] Dataset __getitem__ structure and shapes verified.")

    # --- Test DataLoader ---
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    print("    [Pass] DataLoader batch generation verified.")

    # --- Test Model ---
    device = Config.DEVICE
    model = CactusModel(
        arch="RepVGG", in_chans=3, num_classes=1, use_film=True, use_mtl=True
    ).to(device)

    # Move batch to device
    imgs = batch["image"].to(device)
    fs_norm = batch["file_size_norm"].to(device)

    # Forward Pass
    output = model(imgs, file_size_norm=fs_norm)

    # Assertions for Model
    assert "logits" in output
    assert "mtl_pred" in output
    assert output["logits"].shape == (
        8,
        1,
    ), f"Expected logits shape (8, 1), got {output['logits'].shape}"
    assert output["mtl_pred"].shape == (
        8,
        1,
    ), f"Expected mtl_pred shape (8, 1), got {output['mtl_pred'].shape}"

    print("    [Pass] Model forward pass and output shapes verified.\n")


# =============================================================================
# 3. Integration Testing: Full Stacking Pipeline
# =============================================================================
def run_pipeline_demo():
    """
    Runs the StackingTrainer with truncated data to demonstrate the full workflow.
    """
    print(">>> Starting Integration Test: Stacking Pipeline...")

    trainer = StackingTrainer()

    # 1. Load Data
    # This loads the real data from ./input via metadata
    trainer.load_all_data()

    # 2. Truncate Data for Speed
    # We slice the loaded arrays to a very small number (e.g., 200 samples)
    # so that 2-fold CV finishes almost instantly.
    limit = 200
    print(
        f"    [Info] Truncating training data from {len(trainer.train_imgs)} to {limit} samples for demo."
    )

    trainer.train_imgs = trainer.train_imgs[:limit]
    trainer.train_labels = trainer.train_labels[:limit]
    trainer.train_fsizes = trainer.train_fsizes[:limit]
    trainer.train_ids = trainer.train_ids[:limit]

    # Also truncate test data
    test_limit = 50
    trainer.test_imgs = trainer.test_imgs[:test_limit]
    trainer.test_fsizes = trainer.test_fsizes[:test_limit]
    trainer.test_ids = trainer.test_ids[:test_limit]

    # 3. Run Base Models (Level 0)
    # This will train RepVGG on 2 folds for 1 epoch each
    print("    [Step] Training Base Models (Level 0)...")
    oof_df, test_preds_df = trainer.get_base_model_predictions(load_cached_data=False)

    assert len(oof_df) == limit, "OOF DataFrame length mismatch."
    assert len(test_preds_df) == test_limit, "Test Preds DataFrame length mismatch."
    assert "RepVGG_Spatial" in oof_df.columns, "Model predictions missing from OOF."
    print("    [Pass] Base model training complete.")

    # 4. Train Meta Learner (Level 1)
    print("    [Step] Training Meta Learner (Level 1)...")
    meta_model = trainer.train_meta_learner(oof_df)
    assert meta_model is not None
    print("    [Pass] Meta learner trained.")

    # 5. Generate Submission
    print("    [Step] Generating Submission...")
    trainer.generate_submission(meta_model, test_preds_df)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify Submission Format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == ["id", "has_cactus"]
    assert len(sub_df) == test_limit
    print(
        f"    [Pass] Submission generated at {submission_path} with {len(sub_df)} rows."
    )

    print("\n>>> Integration Test Complete: Pipeline executed successfully.")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Unit Tests
    test_dataset_and_model()

    # 3. Integration Pipeline
    run_pipeline_demo()

    print("\nAll demonstrations passed successfully.")
