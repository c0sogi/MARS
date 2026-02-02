import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, setup_logger
from library.dataset import load_cached_data, get_dataloaders
from library.model import AngleGatedResNet
from library.engine import Engine
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("============================================================")
    print("      Iceberg Classification Pipeline Demonstration")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # ------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # We override the Config class attributes directly to speed up the run
    Config.WORKING_DIR = "./working/demo_run"
    Config.MAX_EPOCHS = 1  # Train for only 1 epoch per fold
    Config.SWA_EPOCHS = 1  # Run SWA for only 1 epoch
    Config.N_FOLDS = 2  # Run only 2 folds for calibration instead of 5
    Config.BATCH_SIZE = 32  # Reasonable batch size
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead in short run

    # Ensure working directory exists and is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set global seed
    set_seed(Config.SEED)

    # Setup logger
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "demo.log"))
    logger.info("Configuration overrides applied for demo.")

    # ------------------------------------------------------------------
    # 2. Data Processing & Loading
    # ------------------------------------------------------------------
    print("\n[Step 2] Processing and Loading Data...")

    # Load training data (this will trigger processing from json since cache is empty)
    images, angles, labels, ids = load_cached_data(is_train=True, load_cache=True)

    # Validation of Raw Data Shapes
    print(f"   -> Loaded {len(images)} training samples.")
    assert len(images) == len(angles) == len(labels) == len(ids)
    assert images.shape[1:] == (
        75,
        75,
        3,
    ), f"Expected (75, 75, 3), got {images.shape[1:]}"
    assert not np.isnan(images).any(), "Images contain NaN values"
    assert not np.isnan(angles).any(), "Angles contain NaN values"

    # Load test data
    test_images, test_angles, _, test_ids = load_cached_data(
        is_train=False, load_cache=True
    )
    print(f"   -> Loaded {len(test_images)} test samples.")
    assert test_images.shape[1:] == (75, 75, 3)

    # ------------------------------------------------------------------
    # 3. DataLoader & Transform Verification
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying DataLoader and Transforms...")

    # Get calibration dataloaders (Fold 0)
    train_loader, val_loader = get_dataloaders(
        fold=0, phase="calibration", load_cache=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    b_imgs = batch["image"]
    b_angs = batch["inc_angle"]
    b_lbls = batch["label"]

    # Validate Batch Shapes (Transforms resize to Config.IMAGE_SIZE = 224)
    expected_shape = (Config.BATCH_SIZE, 3, 224, 224)
    print(f"   -> Batch Image Shape: {b_imgs.shape}")
    assert (
        b_imgs.shape == expected_shape
    ), f"Expected {expected_shape}, got {b_imgs.shape}"
    assert b_angs.shape == (
        Config.BATCH_SIZE,
    ), f"Expected ({Config.BATCH_SIZE},), got {b_angs.shape}"
    assert b_lbls.shape == (
        Config.BATCH_SIZE,
    ), f"Expected ({Config.BATCH_SIZE},), got {b_lbls.shape}"

    print("   -> DataLoader verification passed.")

    # ------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # ------------------------------------------------------------------
    print("\n[Step 4] Initializing AngleGatedResNet...")

    device = torch.device(Config.DEVICE)
    model = AngleGatedResNet().to(device)

    # Run a forward pass with the batch fetched earlier
    b_imgs = b_imgs.to(device)
    b_angs = b_angs.to(device)

    with torch.no_grad():
        logits = model(b_imgs, b_angs)

    print(f"   -> Output Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch."
    print("   -> Forward pass successful.")

    # ------------------------------------------------------------------
    # 5. Phase 1: Calibration (Global Epoch Selection)
    # ------------------------------------------------------------------
    print("\n[Step 5] Running Phase 1: Calibration (Finding Optimal Epoch)...")

    # This runs CV (2 folds as configured) to find the best epoch
    optimal_epoch = Engine.find_optimal_epoch()

    print(f"   -> Phase 1 finished. Optimal Epoch determined: {optimal_epoch}")
    assert isinstance(
        optimal_epoch, (int, np.integer)
    ), "Optimal epoch must be an integer"
    assert optimal_epoch > 0, "Optimal epoch must be positive"

    # ------------------------------------------------------------------
    # 6. Phase 2: Production (Full Fit + SWA)
    # ------------------------------------------------------------------
    print("\n[Step 6] Running Phase 2: Production (Training SWA Models)...")

    # Train 2 independent models on full data for demonstration
    num_models_demo = 2
    Engine.train_full_fit_swa(optimal_epoch, num_models=num_models_demo)

    # Verify checkpoints exist
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    assert os.path.exists(checkpoint_dir), "Checkpoint directory not created."

    for i in range(num_models_demo):
        ckpt_path = os.path.join(checkpoint_dir, f"swa_model_{i}.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} missing."
        print(f"   -> Verified checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # 7. Inference & Submission Generation
    # ------------------------------------------------------------------
    print("\n[Step 7] Generating Submission...")

    generate_submission(num_models=num_models_demo)

    submission_path = os.path.join(Config.WORKING_DIR, "submission", "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated."

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"   -> Submission shape: {df_sub.shape}")

    # Check against sample submission length (from metadata/test_metadata.csv or raw file)
    # We can check against the loaded test_ids
    assert len(df_sub) == len(
        test_ids
    ), f"Submission length {len(df_sub)} != Test set size {len(test_ids)}"

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Incorrect columns in submission."

    # Check value range
    assert (
        df_sub["is_iceberg"].min() >= 0.0 and df_sub["is_iceberg"].max() <= 1.0
    ), "Probabilities out of bounds."

    print("   -> Submission verified successfully.")
    print("\n============================================================")
    print("      Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
