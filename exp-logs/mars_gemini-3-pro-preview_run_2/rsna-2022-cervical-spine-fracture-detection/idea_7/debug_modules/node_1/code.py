import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, competition_metric, get_logger
from library.data import CervicalSpineDataset, get_bbox_data, get_transforms
from library.model import CervicalSpineModel
from library.loss import HybridLoss
from library.engine import fit, predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Cervical Spine Fracture Detection Demo ===")

    # 1. Setup and Configuration Override for Speed
    # We modify the global Config class to run a lightweight version of the task
    seed_everything(42)

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SEQ_LEN = 16  # Reduced from 96 to 16 for speed
    Config.IMAGE_SIZE = (256, 256)  # Reduced from 384 to 256 for speed
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "output/submission.csv")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, SeqLen={Config.SEQ_LEN}"
    )

    # 2. Data Preparation (Subsampling)
    print("\n[Data] Preparing datasets...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subsample data to ensure the script completes quickly
    # Taking enough for 2 batches
    subset_size = Config.BATCH_SIZE * 2
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    # Load Bounding Box Map (force processing to verify logic)
    bbox_map = get_bbox_data(load_cached_data=False)

    # Instantiate Datasets with subsets
    train_ds = CervicalSpineDataset(
        train_subset, bbox_map, transform=get_transforms("train"), mode="train"
    )
    val_ds = CervicalSpineDataset(
        val_subset, bbox_map, transform=get_transforms("val"), mode="val"
    )
    test_ds = CervicalSpineDataset(
        test_subset, bbox_map, transform=get_transforms("test"), mode="test"
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Train batches: {len(train_loader)}")

    # 3. Model Initialization & Forward Pass Check
    print("\n[Model] Initializing and checking forward pass...")
    device = Config.DEVICE
    model = CervicalSpineModel().to(device)

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["images"].to(device)
    study_targets = batch["study_targets"].to(device)
    slice_targets = batch["slice_targets"].to(device)
    slice_mask = batch["slice_mask"].to(device)

    print(f"Input Images: {images.shape}")

    # Perform forward pass
    study_logits, slice_logits = model(images)

    print(f"Output Study Logits: {study_logits.shape}")
    print(f"Output Slice Logits: {slice_logits.shape}")

    # Assertions
    assert study_logits.shape == (Config.BATCH_SIZE, 8), "Study logits shape mismatch"
    assert slice_logits.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Slice logits shape mismatch"

    # 4. Loss Calculation Check
    print("\n[Loss] Verifying loss calculation...")
    criterion = HybridLoss().to(device)
    loss = criterion(
        study_logits, study_targets, slice_logits, slice_targets, slice_mask
    )

    print(f"Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss must be positive"

    # 5. Training Loop
    print("\n[Training] Starting training loop...")
    # fit() handles the loop, validation, and saving the model
    fit(model, train_loader, val_loader, epochs=Config.EPOCHS, device=device)

    # Verify model was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
    print(f"Model successfully saved to {Config.MODEL_PATH}")

    # 6. Prediction
    print("\n[Inference] Generating predictions...")
    # predict() loads the best model and generates submission.csv
    predict(model, test_loader, device=device)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # 7. Metric Verification
    print("\n[Metric] Verifying competition metric...")
    # Dummy data: 2 samples, 8 classes
    y_true_dummy = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]])
    y_pred_dummy = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
        ]
    )

    score = competition_metric(y_pred_dummy, y_true_dummy)
    print(f"Metric Score on dummy data: {score:.4f}")
    assert score > 0, "Metric calculation failed."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
