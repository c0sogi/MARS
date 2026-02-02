import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config, seed_everything
from library.dataset import VinBigDataDataset
from library.model import MultiTaskCenterNet
from library.loss import MultiTaskLoss
from library.engine import run_training
from library.inference import predict_and_format


def main():
    print(">>> Starting Thoracic Disease Detection Pipeline Demo")

    # 1. Configuration & Setup
    # Override Config for a fast demonstration run
    seed_everything(42)

    Config.DEBUG = True
    Config.DEBUG_SIZE = 32  # Use a small subset for training/validation
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Define a specific working directory for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Verification
    print("\n>>> Verifying Dataset Pipeline...")
    # Initialize datasets
    train_ds = VinBigDataDataset(
        split="train", debug=True, debug_size=Config.DEBUG_SIZE
    )
    val_ds = VinBigDataDataset(split="val", debug=True, debug_size=Config.DEBUG_SIZE)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")

    # Verify a single sample
    img, target = train_ds[0]

    # Check Image Shape [3, 512, 512]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"

    # Check Target Shapes (Heatmap stride is 4, so 512/4 = 128)
    feat_size = Config.IMG_SIZE // 4
    assert target["hm"].shape == (
        Config.NUM_CLASSES,
        feat_size,
        feat_size,
    ), f"Expected heatmap shape ({Config.NUM_CLASSES}, {feat_size}, {feat_size}), got {target['hm'].shape}"

    print("Dataset verification passed.")

    # 3. Model & Loss Verification
    print("\n>>> Verifying Model and Loss...")
    model = MultiTaskCenterNet().to(Config.DEVICE)
    criterion = MultiTaskLoss()

    # Create a dummy batch from the sample
    dummy_imgs = img.unsqueeze(0).to(Config.DEVICE)  # [1, 3, 512, 512]

    # Prepare dummy targets (add batch dimension)
    dummy_targets = {}
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            dummy_targets[k] = v.unsqueeze(0).to(Config.DEVICE)
        else:
            dummy_targets[k] = [v]  # List for non-tensor metadata

    # Forward Pass
    model.train()
    outputs = model(dummy_imgs)

    # Verify Output Keys
    required_keys = ["hm", "wh", "reg", "global_logits"]
    for k in required_keys:
        assert k in outputs, f"Model output missing key: {k}"

    # Verify Output Shapes
    assert outputs["hm"].shape == (1, Config.NUM_CLASSES, feat_size, feat_size)
    assert outputs["global_logits"].shape == (1, 1)

    # Loss Calculation
    loss, loss_stats = criterion(outputs, dummy_targets)
    print(f"Initial Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    print("Model and Loss verification passed.")

    # 4. Training Loop
    print("\n>>> Starting Training (1 Epoch)...")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run the engine
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
    )

    # Verify checkpoint creation
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    print("Training completed and model saved.")

    # 5. Inference Pipeline
    print("\n>>> Starting Inference...")

    # Note: predict_and_format creates its own test dataset instance.
    # It will run on the full test set (1500 images).
    # Since we are on A100, this is fast enough (< 1 min).

    # We pass the explicit model path because default args in predict_and_format
    # might have been bound before we modified Config.MODEL_SAVE_PATH.
    predict_and_format(
        model_path=Config.MODEL_SAVE_PATH,
        batch_size=Config.BATCH_SIZE,  # Use small batch size defined above
        device=Config.DEVICE,
        num_workers=Config.NUM_WORKERS,
    )

    # 6. Result Validation
    print("\n>>> Validating Submission...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print("Head:")
    print(df_sub.head())

    # Basic checks
    assert "image_id" in df_sub.columns
    assert "PredictionString" in df_sub.columns
    assert len(df_sub) == 1500, f"Expected 1500 predictions, got {len(df_sub)}"

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
