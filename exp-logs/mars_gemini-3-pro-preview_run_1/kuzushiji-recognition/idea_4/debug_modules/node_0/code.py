import sys
import os
import torch
import pandas as pd
import numpy as np

# Append current directory to system path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, load_metadata
from library.dataset import KuzushijiDataset
from library.model import CenterNetConvNeXt
from library.loss import CenterNetLoss
from library.trainer import Trainer
from library.inference import Predictor


def run_demonstration():
    print("===========================================================")
    print("   Kuzushiji Recognition Pipeline Demonstration")
    print("===========================================================")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config to run quickly on the available hardware
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.IMG_SIZE = 512  # Reduced image size (512x512) for speed
    Config.DEBUG = True  # Enable debug mode
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for training/val
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in demo
    Config.CONF_THRESHOLD = 0.05  # Low threshold to ensure some predictions are made

    # Apply reproducibility settings with new config
    seed_everything(Config.SEED)
    print("    Configuration updated: 1 Epoch, 512px Images, Debug Mode (20 samples).")

    # ---------------------------------------------------------
    # 2. Dataset & Data Loading Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset and Data Loading...")

    # Load metadata manually to test dataset class
    train_meta = load_metadata(Config.TRAIN_METADATA_PATH)
    # Slice manually for this specific test, though Trainer handles it internally too
    train_meta_subset = train_meta.head(Config.DEBUG_SAMPLE_SIZE).copy()

    # Instantiate Dataset
    dataset = KuzushijiDataset(train_meta_subset, mode="train", load_cached_data=False)
    print(f"    Dataset initialized with {len(dataset)} samples.")

    # Fetch one sample
    sample = dataset[0]
    img_tensor = sample["image"]
    hm_tensor = sample["hm"]

    # Check shapes
    # Image: (3, H, W)
    expected_img_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    # Heatmap: (1, H/4, W/4)
    expected_hm_shape = (1, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4)

    assert (
        img_tensor.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {img_tensor.shape}"
    assert (
        hm_tensor.shape == expected_hm_shape
    ), f"Heatmap shape mismatch. Expected {expected_hm_shape}, got {hm_tensor.shape}"

    print("    Dataset shapes verified successfully.")

    # ---------------------------------------------------------
    # 3. Model & Loss Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture and Loss Calculation...")

    # Update class count based on dataset
    Config.NUM_CLASSES = dataset.num_classes
    print(f"    Number of classes detected: {Config.NUM_CLASSES}")

    # Initialize Model
    model = CenterNetConvNeXt(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)
    model.train()

    # Prepare a dummy batch on device
    batch = {
        "image": img_tensor.unsqueeze(0).to(Config.DEVICE),
        "hm": sample["hm"].unsqueeze(0).to(Config.DEVICE),
        "wh": sample["wh"].unsqueeze(0).to(Config.DEVICE),
        "reg": sample["reg"].unsqueeze(0).to(Config.DEVICE),
        "ind": sample["ind"].unsqueeze(0).to(Config.DEVICE),
        "cat": sample["cat"].unsqueeze(0).to(Config.DEVICE),
        "reg_mask": sample["reg_mask"].unsqueeze(0).to(Config.DEVICE),
    }

    # Forward Pass
    outputs = model(batch["image"])

    # Verify Output Keys and Shapes
    assert "hm" in outputs and "wh" in outputs and "reg" in outputs and "cls" in outputs
    assert outputs["hm"].shape == (1, 1, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4)
    assert outputs["cls"].shape == (
        1,
        Config.NUM_CLASSES,
        Config.IMG_SIZE // 4,
        Config.IMG_SIZE // 4,
    )

    # Calculate Loss
    criterion = CenterNetLoss()
    loss, loss_stats = criterion(outputs, batch)

    print(f"    Forward pass successful. Total Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[4] Executing Training Loop (Trainer)...")

    # Initialize Trainer
    # debug=True tells Trainer to slice the dataframes to Config.DEBUG_SAMPLE_SIZE
    trainer = Trainer(debug=True)

    # Run training
    trainer.fit()

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"    Training complete. Checkpoint saved to {checkpoint_path}")

    # ---------------------------------------------------------
    # 5. Inference Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Inference Pipeline...")

    # Initialize Predictor
    predictor = Predictor(checkpoint_name="best_model.pth")

    # Define output path
    submission_file = "./submission/demo_submission.csv"

    # Run Inference
    # Note: Predictor loads the full test set metadata.
    # Since the test set is small (361 images), we can run full inference quickly.
    predictor.run(output_path=submission_file)

    # Verify Submission File
    assert os.path.exists(submission_file), "Submission file was not created."

    df_sub = pd.read_csv(submission_file)
    print(f"    Submission file created with {len(df_sub)} rows.")

    # Check columns
    assert "image_id" in df_sub.columns and "labels" in df_sub.columns

    # Check content of first row
    if len(df_sub) > 0:
        print(f"    Sample prediction: {df_sub.iloc[0]['labels'][:50]}...")

    print("\n===========================================================")
    print("   Demonstration Completed Successfully")
    print("===========================================================")


if __name__ == "__main__":
    # Ensure working directory exists for logs/checkpoints
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    run_demonstration()
