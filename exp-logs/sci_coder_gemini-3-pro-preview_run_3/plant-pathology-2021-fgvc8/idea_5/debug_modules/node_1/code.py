import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for imports
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score, get_ema_model
from library.dataset import get_loaders
from library.model import AppleDiseaseModel
from library.engine import train_one_epoch, validate


def run_demo():
    print("=== Starting Apple Disease Detection Demo ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast demonstration
    print("Configuring for debug mode...")
    Config.DEBUG = True  # Use a small subset of data (100 samples)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.NUM_WORKERS = 2  # Reduce worker overhead
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir for demo
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[1/4] Verifying Data Pipeline...")

    # Initialize DataLoaders
    # force reload to ensure processing logic runs
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches:   {len(val_loader)}")
    print(f"  Test Batches:  {len(test_loader)}")

    # Fetch a single batch to inspect
    images, targets, image_ids = next(iter(train_loader))

    # Verify Shapes
    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Target Shape: {targets.shape}")

    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    # Verify Data Types
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.float32, "Targets should be float32"

    print("  Data Pipeline Verified.")

    # ==========================================
    # 3. Model Initialization & Logic
    # ==========================================
    print("\n[2/4] Verifying Model Architecture...")

    # Initialize Model
    model = AppleDiseaseModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # Initialize EMA
    ema_model = get_ema_model(model)
    if Config.USE_EMA:
        print("  EMA Model initialized.")
        assert ema_model is not None

    # Forward Pass Check
    model.eval()
    with torch.no_grad():
        dummy_input = images.to(device)
        outputs = model(dummy_input)

    print(f"  Model Output Shape: {outputs.shape}")
    assert (
        outputs.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {outputs.shape}"

    print("  Model Architecture Verified.")

    # ==========================================
    # 4. Training Loop Simulation
    # ==========================================
    print("\n[3/4] Verifying Training & Validation Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for 1 Epoch
    print("  Training for 1 epoch...")
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=0,
        ema_model=ema_model,
    )
    print(f"  Train Loss: {train_loss:.6f}")

    # Basic sanity checks on loss
    assert not np.isnan(train_loss), "Training loss returned NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Validate
    print("  Validating...")
    val_loss, val_score = validate(model, val_loader, device)
    print(f"  Val Loss: {val_loss:.6f}")
    print(f"  Val F1 Score: {val_score:.6f}")

    assert not np.isnan(val_loss), "Validation loss returned NaN"
    assert 0.0 <= val_score <= 1.0, "F1 Score must be between 0 and 1"

    # Save checkpoint (simulating 'fit' function behavior)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"  Checkpoint saved to {Config.BEST_MODEL_PATH}")

    print("  Training Loop Verified.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[4/4] Verifying Inference & Submission Generation...")

    # Load Best Model (conceptually, here we just use the current model)
    model.eval()

    predictions = []
    image_ids_list = []

    with torch.no_grad():
        for imgs, _, ids in test_loader:
            imgs = imgs.to(device)
            # Forward pass
            logits = model(imgs)
            # Sigmoid for multi-label probability
            probs = torch.sigmoid(logits)

            predictions.append(probs.cpu().numpy())
            image_ids_list.extend(ids)

    predictions = np.concatenate(predictions, axis=0)

    # Thresholding
    pred_binary = (predictions > Config.CONF_THRESHOLD).astype(int)

    # Convert binary matrix back to string labels
    submission_rows = []
    for idx, row in enumerate(pred_binary):
        img_id = image_ids_list[idx]
        indices = np.where(row == 1)[0]

        if len(indices) == 0:
            # If no class exceeds threshold, 'healthy' is often the default fallback
            # or the model should have predicted 'healthy' explicitly if it's a class.
            # In this dataset, 'healthy' is a specific class (index 2 in Config.LABELS).
            # If the model is weak (untrained), it might predict nothing.
            label_str = "healthy"
        else:
            labels = [Config.ID2LABEL[i] for i in indices]
            label_str = " ".join(labels)

        submission_rows.append({"image": img_id, "labels": label_str})

    submission_df = pd.DataFrame(submission_rows)

    # Check output
    print("  Sample Submission Data:")
    print(submission_df.head(3))

    # Assertions
    assert len(submission_df) == len(
        test_loader.dataset
    ), "Submission length does not match test set size"
    assert (
        "image" in submission_df.columns and "labels" in submission_df.columns
    ), "Submission columns missing"

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"  Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
