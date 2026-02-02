import os
import pandas as pd
import torch
import numpy as np
import warnings
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.model import HPIRVN
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a demo environment by creating a separate working directory
    and creating subsampled metadata files to speed up processing.
    """
    print("Setting up demo environment...")

    # Define and create a specific working directory for this run
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config to use this directory
    Config.WORKING_DIR = demo_working_dir

    # --- Create Subsampled Metadata ---
    # We read the original metadata and select a few game_plays.
    # This allows us to use the real heavy data processing pipeline but on a tiny subset.

    # 1. Train Metadata Subset (2 plays)
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    train_plays = df_train["game_play"].unique()[:2]
    df_train_sub = df_train[df_train["game_play"].isin(train_plays)].copy()
    train_meta_path = os.path.join(demo_working_dir, "train_meta.csv")
    df_train_sub.to_csv(train_meta_path, index=False)

    # 2. Validation Metadata Subset (1 play)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    val_plays = df_val["game_play"].unique()[:1]
    df_val_sub = df_val[df_val["game_play"].isin(val_plays)].copy()
    val_meta_path = os.path.join(demo_working_dir, "val_meta.csv")
    df_val_sub.to_csv(val_meta_path, index=False)

    # 3. Test Metadata Subset (1 play)
    df_test = pd.read_csv(Config.TEST_META_PATH)
    test_plays = df_test["game_play"].unique()[:1]
    df_test_sub = df_test[df_test["game_play"].isin(test_plays)].copy()
    test_meta_path = os.path.join(demo_working_dir, "test_meta.csv")
    df_test_sub.to_csv(test_meta_path, index=False)

    print(
        f"Subsampled Data: Train={len(df_train_sub)}, Val={len(df_val_sub)}, Test={len(df_test_sub)}"
    )

    # --- Override Config Paths ---
    Config.TRAIN_META_PATH = train_meta_path
    Config.VAL_META_PATH = val_meta_path
    Config.TEST_META_PATH = test_meta_path

    # --- Override Training Hyperparameters for Speed ---
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 256  # Smaller batch size for small dataset
    Config.NUM_WORKERS = 2  # Reduce overhead
    Config.SUBMISSION_DIR = demo_working_dir  # Save submission to working dir


def run_pipeline():
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading (Train/Val)
    print("\n[1/4] Loading and Processing Data...")
    # load_cached_data=False ensures we run the DataProcessor logic from scratch on our new subset
    train_loader, _ = get_dataloader(mode="train", load_cached_data=False)
    val_loader, _ = get_dataloader(mode="validation", load_cached_data=False)

    # Verify Data Integrity
    batch = next(iter(train_loader))
    print(f"  Batch Keys: {list(batch.keys())}")
    print(f"  Geometry Input Shape: {batch['geometry'].shape}")

    assert batch["geometry"].dim() == 2
    assert batch["label"].dim() == 1

    # 3. Model Initialization
    print("\n[2/4] Initializing Model...")
    device = Config.DEVICE
    model = HPIRVN().to(device)

    # Verify Forward Pass
    with torch.no_grad():
        geo = batch["geometry"].to(device)
        mot = batch["motion"].to(device)
        dyn = batch["dynamics"].to(device)
        vis = batch["visual"].to(device)
        logits = model(geo, mot, dyn, vis)

    print(f"  Model Output Shape: {logits.shape}")
    assert logits.shape == (batch["geometry"].shape[0], 1)

    # 4. Training Loop
    print("\n[3/4] Training Model...")
    trainer = Trainer(device=device)

    # The trainer initializes its own model, so we fit that one.
    best_threshold = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    print(f"  Best Threshold Found: {best_threshold:.4f}")

    # Verify Artifacts
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not saved."

    # 5. Inference and Submission
    print("\n[4/4] Generating Submission...")
    test_loader, test_ids = get_dataloader(mode="test", load_cached_data=False)

    trainer.generate_submission(test_loader, test_ids, best_threshold)

    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not generated."

    # Verify Submission Content
    df_sub = pd.read_csv(submission_file)
    print(f"  Submission Head:\n{df_sub.head()}")
    assert len(df_sub) == len(test_ids)
    assert df_sub["contact"].dtype == np.int64 or df_sub["contact"].dtype == int

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    setup_demo_environment()
    run_pipeline()
