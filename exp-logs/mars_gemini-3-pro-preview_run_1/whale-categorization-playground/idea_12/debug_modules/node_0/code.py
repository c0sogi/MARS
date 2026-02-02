import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import get_dataloaders, get_test_loader
from library.models import WhaleModel
from library.loss import ArcFaceLoss
from library.trainer import Trainer
from library.inference import predict_ensemble


def run_demo():
    print("Starting Whale Identification Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and demo purposes
    Config.MAX_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.IMG_SIZE = 128  # Smaller image size for faster processing

    # Define a specific working directory for this demo
    demo_dir = "./working/demo_execution"
    Config.WORKING_DIR = demo_dir
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update Ensemble Config to train/use just one model
    # We use a unique name so the trainer saves 'demo_model_best.pth'
    Config.ENSEMBLE_MODELS = [
        {"arch": "densenet121", "seed": 42, "name": "demo_checkpoint"}
    ]

    # Set global seed
    seed_everything(42)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load a tiny subset of data (debug_size=20)
    train_loader, val_loader, classes = get_dataloaders(
        load_cached_data=False, debug_size=20
    )

    print(f"Number of classes in subset: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Train Batch
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert len(classes) > 0, "No classes found"

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    device = Config.DEVICE
    num_classes = len(classes)

    # Instantiate Model
    model = WhaleModel(
        arch="densenet121", num_classes=num_classes, pretrained=False
    )  # False for speed
    model.to(device)
    model.eval()

    # Dummy Input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    # Forward Pass
    with torch.no_grad():
        embeddings = model(dummy_input)

    print(f"Embedding Output Shape: {embeddings.shape}")
    assert embeddings.shape == (2, Config.EMBEDDING_SIZE), "Model output shape mismatch"

    # Instantiate Loss
    criterion = ArcFaceLoss(
        num_classes=num_classes, embedding_size=Config.EMBEDDING_SIZE
    )
    criterion.to(device)

    # Loss Calculation Check
    dummy_labels = torch.tensor([0, 1]).to(device)
    # Ensure dummy labels are within valid range
    if num_classes < 2:
        dummy_labels = torch.zeros(2, dtype=torch.long).to(device)

    loss = criterion(embeddings, dummy_labels)
    print(f"Calculated Dummy Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    model_cfg = Config.ENSEMBLE_MODELS[0]

    # Re-initialize model for training
    model = WhaleModel(
        arch=model_cfg["arch"], num_classes=num_classes, pretrained=True
    )  # Use pretrained for realism
    model.to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        model_name=model_cfg["name"],
    )

    best_score = trainer.train_until_convergence()
    print(f"Training completed. Best MAP@5: {best_score:.4f}")

    # Verify Checkpoint Creation
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_cfg['name']}_best.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print(f"Checkpoint verified at: {ckpt_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference...")

    # The predict_ensemble function uses Config.ENSEMBLE_MODELS to load models.
    # Since we set that to our single demo model, it should load the checkpoint we just made.
    # Note: predict_ensemble loads the test set internally.

    predict_ensemble()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print("First 3 rows:")
    print(df_sub.head(3))

    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    # -------------------------------------------------------------------------
    # 6. Metric Logic Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Metric Logic (MAP@5)...")

    # Case 1: Perfect prediction
    preds_1 = [["w_1", "w_2", "w_3", "w_4", "w_5"]]
    targets_1 = ["w_1"]
    score_1 = calculate_map5(preds_1, targets_1)
    assert abs(score_1 - 1.0) < 1e-6, f"Expected 1.0, got {score_1}"

    # Case 2: Target at rank 2 (index 1) -> Score 1/2
    preds_2 = [["w_2", "w_1", "w_3", "w_4", "w_5"]]
    targets_2 = ["w_1"]
    score_2 = calculate_map5(preds_2, targets_2)
    assert abs(score_2 - 0.5) < 1e-6, f"Expected 0.5, got {score_2}"

    # Case 3: Target not in top 5 -> Score 0
    preds_3 = [["w_2", "w_3", "w_4", "w_5", "w_6"]]
    targets_3 = ["w_1"]
    score_3 = calculate_map5(preds_3, targets_3)
    assert abs(score_3 - 0.0) < 1e-6, f"Expected 0.0, got {score_3}"

    print("Metric logic verified.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
