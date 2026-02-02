import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import provided library modules
from library.config import Config, seed_everything
from library.utils import AverageMeter, apk, mapk
from library.loss import ArcFaceLoss
from library.model import WhaleModel
from library.dataset import get_train_val_loaders, get_test_loader, get_label_mapping
from library.train import run_training_phase
from library.inference import generate_submission

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("Initializing Demo...")
    seed_everything(42)

    # Define a separate working directory for this demo run
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config paths to use this directory
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 2. Create Mini Dataset (Speed Optimization)
    # -------------------------------------------------------------------------
    print("Creating mini datasets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample Training Data (Ensure we have known whales)
    # Filter out new_whale to ensure we have valid classes for training
    known_train = orig_train[orig_train["Id"] != "new_whale"]
    # Take top 20 images (ensuring we don't exceed available data)
    mini_train = known_train.head(20).copy()

    # Sample Validation and Test Data
    mini_val = orig_val.head(10).copy()
    mini_test = orig_test.head(10).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Calculate number of classes in this mini set
    unique_ids = mini_train["Id"].nunique()
    print(f"Mini Train: {len(mini_train)} samples, {unique_ids} unique classes.")

    # -------------------------------------------------------------------------
    # 3. Apply Config Overrides
    # -------------------------------------------------------------------------
    # Point Config to mini files
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    # Adjust Training Parameters for Speed
    Config.N_CLASSES = unique_ids
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Use a very small resolution and 1 epoch
    Config.STAGES = [{"resolution": 128, "epochs": 1}]

    # Use a single model in the ensemble to save time
    # We use a smaller embedding size for the demo
    Config.MODEL_CONFIGS = [
        {"backbone": "tf_efficientnet_b4", "name": "demo_model", "embedding_size": 128}
    ]

    # -------------------------------------------------------------------------
    # 4. Verify Utilities
    # -------------------------------------------------------------------------
    print("\nVerifying Utilities...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)
    # Total sum = 10*2 + 20*1 = 40. Total count = 3. Avg = 13.33
    assert abs(meter.avg - 13.333) < 0.01, "AverageMeter failed"

    # Test APK/MAPK
    # Perfect prediction
    score = apk(["w_1"], ["w_1", "w_2"], k=5)
    assert score == 1.0, "APK failed for perfect match"
    # No match
    score = apk(["w_1"], ["w_2", "w_3"], k=5)
    assert score == 0.0, "APK failed for no match"
    print("Utilities Verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Model & Loss
    # -------------------------------------------------------------------------
    print("\nVerifying Model and Loss...")

    device = Config.DEVICE

    # Instantiate Model
    model = WhaleModel(
        model_name=Config.MODEL_CONFIGS[0]["backbone"],
        pretrained=False,  # False for speed/safety in demo, Train loop uses True
        embedding_size=Config.MODEL_CONFIGS[0]["embedding_size"],
    ).to(device)

    # Create dummy input (Batch=2, C=3, H=128, W=128)
    dummy_input = torch.randn(2, 3, 128, 128).to(device)

    # Forward Pass
    embeddings = model(dummy_input)
    assert embeddings.shape == (
        2,
        128,
    ), f"Model output shape mismatch: {embeddings.shape}"

    # Instantiate Loss
    loss_fn = ArcFaceLoss(
        in_features=128, out_features=Config.N_CLASSES, s=30.0, m=0.5
    ).to(device)

    # Create dummy labels
    dummy_labels = torch.tensor([0, min(1, Config.N_CLASSES - 1)]).to(device)

    # Calculate Loss
    loss = loss_fn(embeddings, dummy_labels)
    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"Model and Loss Verified. Loss value: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 6. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\nVerifying Data Loaders...")

    # This will trigger cache generation in ./working/demo_run
    train_loader, val_loader = get_train_val_loaders(
        resolution=128, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        128,
        128,
    ), "Loader image shape mismatch"
    assert labels.shape == (Config.BATCH_SIZE,), "Loader label shape mismatch"
    print("Data Loaders Verified.")

    # -------------------------------------------------------------------------
    # 7. Run Training Phase
    # -------------------------------------------------------------------------
    print("\nStarting Training Phase Demo...")
    # This runs the full training logic defined in library/train.py
    # using our overridden Config (1 epoch, mini data)
    run_training_phase()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_model_128_best.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"Training completed. Checkpoint found at {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 8. Run Inference Phase
    # -------------------------------------------------------------------------
    print("\nStarting Inference Phase Demo...")
    # This runs the full inference logic defined in library/inference.py
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())

    assert len(df_sub) == len(mini_test), "Submission row count mismatch"
    assert (
        "new_whale" in df_sub.iloc[0]["Id"]
    ), "Prediction string format seems incorrect (should contain whale IDs)"

    print("\nDemo Completed Successfully!")
