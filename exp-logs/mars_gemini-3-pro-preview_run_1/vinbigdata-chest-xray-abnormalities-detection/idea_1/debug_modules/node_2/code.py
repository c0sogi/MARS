import os
import torch
import numpy as np
import pandas as pd
import cv2
import sys

# Import library components
from library.config import Config
from library.dataset import VinBigDataDataset
from library.model import CenterNet
from library.loss import CenterNetLoss
from library.inference import decode_predictions, convert_to_prediction_string
from library.trainer import Trainer, seed_everything


def run_demo():
    print("=== Starting Library Demo ===")

    # 1. Setup & Configuration Override
    # We modify the Config class directly to optimize for a quick demo run.
    print("\n[1] Configuring environment...")
    seed_everything(42)

    # Override Config for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Use only 4 images
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.BACKBONE = "resnet18"  # Lightweight backbone

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Dataset Verification
    print("\n[2] Verifying Dataset...")
    # Initialize dataset in train mode
    dataset = VinBigDataDataset(split="train", debug=True)

    # Fetch one sample
    sample = dataset[0]

    # Verify keys
    expected_keys = ["image", "hm", "wh", "reg", "ind", "reg_mask", "image_id"]
    for key in expected_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify Shapes
    img = sample["image"]
    hm = sample["hm"]
    wh = sample["wh"]

    print(f"  Image Shape: {img.shape}")
    print(f"  Heatmap Shape: {hm.shape}")

    # Assertions
    # Image should be (3, 512, 512) based on Config.IMG_SIZE=512
    assert img.shape == (3, 512, 512), f"Incorrect image shape: {img.shape}"
    # Heatmap should be (Num_Classes, 128, 128) based on Down_Ratio=4
    assert hm.shape == (
        Config.NUM_CLASSES,
        128,
        128,
    ), f"Incorrect heatmap shape: {hm.shape}"
    # WH should be (Max_Objects, 2)
    assert wh.shape == (Config.MAX_OBJECTS, 2), f"Incorrect WH shape: {wh.shape}"

    print("  Dataset verification passed.")

    # 3. Model Verification
    print("\n[3] Verifying Model...")
    # Initialize model (pretrained=False for faster initialization)
    model = CenterNet(pretrained=False)
    model.eval()

    # Create dummy input batch (Batch Size = 2)
    dummy_input = torch.randn(2, 3, 512, 512)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    # Verify output keys
    assert "hm" in outputs
    assert "wh" in outputs
    assert "reg" in outputs

    # Verify output shapes
    # hm: (B, C, H/4, W/4)
    assert outputs["hm"].shape == (2, Config.NUM_CLASSES, 128, 128)
    # wh: (B, 2, H/4, W/4) - Note: Model outputs dense map, dataset returns sparse list
    assert outputs["wh"].shape == (2, 2, 128, 128)

    print("  Model forward pass successful.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function...")
    criterion = CenterNetLoss()

    # Create dummy targets matching the structure of a batch from DataLoader
    # We need to stack the sample tensors to simulate a batch of 2
    batch_targets = {
        "hm": torch.stack([sample["hm"], sample["hm"]]),
        "wh": torch.stack([sample["wh"], sample["wh"]]),
        "reg": torch.stack([sample["reg"], sample["reg"]]),
        "ind": torch.stack([sample["ind"], sample["ind"]]),
        "reg_mask": torch.stack([sample["reg_mask"], sample["reg_mask"]]),
    }

    # Calculate loss
    # outputs["hm"] is raw logits, criterion handles sigmoid
    loss, loss_stats = criterion(outputs, batch_targets)

    print(f"  Total Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"
    assert "hm_loss" in loss_stats
    print("  Loss calculation successful.")

    # 5. Inference Logic Verification
    print("\n[5] Verifying Inference Logic...")
    # Use the dummy outputs from step 3
    # decode_predictions expects hm, wh, reg
    detections = decode_predictions(outputs["hm"], outputs["wh"], outputs["reg"], K=10)

    # Shape should be (Batch, K, 6)
    assert detections.shape == (
        2,
        10,
        6,
    ), f"Incorrect detection shape: {detections.shape}"

    # Test string conversion on the first image's detections
    # We need to move to CPU and numpy
    det_np = detections[0].detach().cpu().numpy()
    pred_str = convert_to_prediction_string(
        det_np, conf_threshold=0.0
    )  # threshold 0 to ensure string content

    print(f"  Sample Prediction String: {pred_str[:50]}...")
    assert isinstance(pred_str, str)
    print("  Inference logic verified.")

    # 6. Trainer Integration (Full Loop)
    print("\n[6] Running Trainer (Train & Predict)...")

    # Initialize Trainer
    # This will load the dataset (using the DEBUG subset defined in Config)
    trainer = Trainer(load_cached_data=False)

    # Train
    # Since NUM_EPOCHS=1 and DEBUG=True, this should be very fast
    print("  Starting training loop...")
    trainer.train()

    # Check if model was saved
    assert os.path.exists(trainer.model_save_path), "Model checkpoint was not created."
    print("  Training completed and model saved.")

    # Predict
    # This generates submission.csv
    print("  Starting inference loop...")
    trainer.predict()

    # Verify submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission rows: {len(df_sub)}")
    print(f"  Submission columns: {list(df_sub.columns)}")

    # Check format
    assert "image_id" in df_sub.columns
    assert "PredictionString" in df_sub.columns
    # Should have rows equal to test set size (or debug sample size if debug applies to test)
    # The Trainer passes Config.DEBUG to test dataset, so it should be small.
    assert len(df_sub) > 0

    print("  Trainer integration successful.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
