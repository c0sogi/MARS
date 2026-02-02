import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
from library.config import Config
from library.utils import seed_everything, pf1_score
from library.dataset import get_dataloaders
from library.model import SiameseFPNEfficientNet
from library.train import run_training
from library.inference import predict_submission


def create_debug_metadata(src_path, dst_path, n_samples=16):
    """
    Creates a small subset of metadata containing only rows where image files exist.
    """
    if not os.path.exists(src_path):
        print(f"Source metadata not found: {src_path}")
        return

    df = pd.read_csv(src_path)
    valid_rows = []

    # Iterate to find rows with existing images
    # We only need a small number of valid samples for the demo
    count = 0
    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(full_path):
            valid_rows.append(row)
            count += 1
            if count >= n_samples:
                break

    if not valid_rows:
        raise FileNotFoundError(
            f"No valid images found for {src_path} in {Config.INPUT_DIR}"
        )

    df_subset = pd.DataFrame(valid_rows)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df_subset.to_csv(dst_path, index=False)
    print(f"Created debug metadata at {dst_path} with {len(df_subset)} rows.")


def setup_debug_environment():
    """
    Overrides Config parameters for a fast, minimal execution.
    """
    print("Setting up debug environment...")

    # 1. Define paths
    debug_dir = "./working/demo_run"
    os.makedirs(debug_dir, exist_ok=True)

    debug_meta_dir = os.path.join(debug_dir, "metadata")

    # 2. Create subset metadata
    # We use the existing metadata files to create tiny subsets
    train_meta_src = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta_src = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta_src = os.path.join(Config.METADATA_DIR, "test.csv")

    train_meta_dst = os.path.join(debug_meta_dir, "train.csv")
    val_meta_dst = os.path.join(debug_meta_dir, "val.csv")
    test_meta_dst = os.path.join(debug_meta_dir, "test.csv")

    create_debug_metadata(train_meta_src, train_meta_dst, n_samples=16)
    create_debug_metadata(val_meta_src, val_meta_dst, n_samples=8)
    create_debug_metadata(test_meta_src, test_meta_dst, n_samples=8)

    # 3. Override Config class attributes
    Config.WORK_DIR = debug_dir
    Config.TRAIN_METADATA = train_meta_dst
    Config.VAL_METADATA = val_meta_dst
    Config.TEST_METADATA = test_meta_dst

    # Update cache paths to avoid conflicts with real run
    Config.CACHE_TRAIN_PATH = os.path.join(debug_dir, "processed_train.parquet")
    Config.CACHE_VAL_PATH = os.path.join(debug_dir, "processed_val.parquet")
    Config.CACHE_TEST_PATH = os.path.join(debug_dir, "processed_test.parquet")
    Config.CACHE_AGE_STATS = os.path.join(debug_dir, "age_stats.npy")

    Config.MODEL_PATH = os.path.join(debug_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(debug_dir, "submission.csv")

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.VAL_BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.IMG_SIZE = (256, 256)  # Reduce image size for speed in demo

    # Use update method for standard flags
    Config.update(debug=True)


def verify_metric():
    print("\n=== Verifying Metric Logic ===")
    # Case 1: Perfect prediction
    labels = np.array([1, 0, 1])
    preds = np.array([1.0, 0.0, 1.0])
    score = pf1_score(labels, preds)
    print(f"Perfect Score: {score}")
    assert np.isclose(score, 1.0), "Metric failed on perfect prediction"

    # Case 2: Zero prediction
    preds_zero = np.array([0.0, 0.0, 0.0])
    score_zero = pf1_score(labels, preds_zero)
    print(f"Zero Score: {score_zero}")
    assert np.isclose(score_zero, 0.0), "Metric failed on zero prediction"

    print("Metric verification passed.")


def verify_dataloader():
    print("\n=== Verifying DataLoader ===")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    batch = next(iter(train_loader))

    # Check keys
    assert "image" in batch
    assert "image_contra" in batch
    assert "label" in batch

    # Check Shapes
    # Image: [B, 3, H, W] (3 channels: Img, Age, Implant)
    imgs = batch["image"]
    B, C, H, W = imgs.shape

    print(f"Batch Size: {B}, Channels: {C}, Height: {H}, Width: {W}")

    assert B == Config.BATCH_SIZE
    assert C == 3  # Image + Age + Implant
    assert H == Config.IMG_SIZE[0]
    assert W == Config.IMG_SIZE[1]

    # Check Contralateral
    contra = batch["image_contra"]
    assert contra.shape == imgs.shape

    print("DataLoader verification passed.")
    return train_loader


def verify_model(loader):
    print("\n=== Verifying Model Architecture ===")
    device = Config.DEVICE
    model = SiameseFPNEfficientNet()
    model.to(device)
    model.train()

    # Get a batch
    batch = next(iter(loader))
    imgs = batch["image"].to(device)
    contra = batch["image_contra"].to(device)
    labels = batch["label"].to(device).unsqueeze(1)

    # Forward Pass
    logits = model(imgs, contra)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1)

    # Backward Pass check
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    loss.backward()

    print("Model forward/backward pass verification passed.")


def run_full_pipeline_demo():
    print("\n=== Running Full Training & Inference Pipeline Demo ===")

    # 1. Training
    # This will use the debug config (1 epoch, tiny dataset)
    best_pf1 = run_training(load_cached_data=False)

    # Verify model was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")
    print("Training complete. Checkpoint verified.")

    # 2. Inference
    submission_df = predict_submission(load_cached_data=False)

    # Verify submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("Inference complete. Submission file verified.")
    print("\nSample Submission Rows:")
    print(submission_df.head())

    # Check columns
    assert "prediction_id" in submission_df.columns
    assert "cancer" in submission_df.columns

    # Check values are probabilities
    assert submission_df["cancer"].min() >= 0.0
    assert submission_df["cancer"].max() <= 1.0


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # Setup environment
    setup_debug_environment()

    # Verify individual components
    verify_metric()
    train_loader = verify_dataloader()
    verify_model(train_loader)

    # Run the main task
    run_full_pipeline_demo()

    print("\nAll demonstrations completed successfully.")
