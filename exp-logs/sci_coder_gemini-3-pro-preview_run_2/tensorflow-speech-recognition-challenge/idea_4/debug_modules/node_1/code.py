import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.audio_transforms import get_transforms
from library.dataset import SpeechCommandDataset, get_dataloaders
from library.model import ConvNeXtAudio
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Speech Command Recognition Pipeline Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configure for Speed (Override Config)
    # ---------------------------------------------------------
    print("[1] Configuring parameters for fast execution...")

    # Enable Debug mode to use a small subset of data (500 samples)
    Config.DEBUG = True

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Disable pretrained weights to avoid download time/errors
    Config.PRETRAINED = False

    # Disable multiprocessing for simple sequential execution
    Config.NUM_WORKERS = 0

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Clean up previous demo runs if any
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Verify Utils
    # ---------------------------------------------------------
    print("\n[2] Verifying Utils module...")
    set_seed(42)

    # Test Checkpointing Logic
    dummy_model = torch.nn.Linear(10, 2)
    dummy_optim = torch.optim.SGD(dummy_model.parameters(), lr=0.01)
    ckpt_path = os.path.join(Config.WORKING_DIR, "test_ckpt.pth")

    save_checkpoint(dummy_model, dummy_optim, epoch=0, loss=0.123, path=ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    loaded = load_checkpoint(ckpt_path, dummy_model, dummy_optim, device="cpu")
    assert loaded["epoch"] == 0
    assert loaded["loss"] == 0.123
    print("    Utils verification passed.")

    # ---------------------------------------------------------
    # 3. Verify Audio Transforms
    # ---------------------------------------------------------
    print("\n[3] Verifying Audio Transforms...")
    transforms = get_transforms(phase="train")

    # Create dummy audio: 1 second of mono audio at 16kHz
    # Shape: (Batch=1, Channels=1, Time=16000)
    dummy_audio = torch.randn(1, 1, 16000)

    # Apply transforms
    spec = transforms(dummy_audio)

    print(f"    Input Audio Shape: {dummy_audio.shape}")
    print(f"    Output Spec Shape: {spec.shape}")

    # Validation
    # Expected: (Batch=1, Channels=1, Freq=64, Time=~101)
    assert spec.dim() == 4, f"Expected 4D tensor, got {spec.dim()}"
    assert spec.shape[1] == 1, "Channel dimension should be 1"
    assert (
        spec.shape[2] == Config.N_MELS
    ), f"Frequency dimension should be {Config.N_MELS}"
    assert spec.shape[3] > 0, "Time dimension should be positive"
    print("    Transforms verification passed.")

    # ---------------------------------------------------------
    # 4. Verify Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n[4] Verifying Dataset and DataLoader...")

    # Instantiate Dataset (Debug mode loads subset)
    train_ds = SpeechCommandDataset(Config.TRAIN_METADATA, phase="train", debug=True)

    print(f"    Dataset size (debug): {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset should not be empty."

    # Check single item retrieval
    feat, label = train_ds[0]
    print(f"    Item 0 Feature Shape: {feat.shape}")
    print(f"    Item 0 Label ID: {label}")

    assert isinstance(feat, torch.Tensor)
    assert isinstance(label, (int, np.integer))
    assert feat.shape[1] == Config.N_MELS

    # Check DataLoader batching
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)
    batch_feat, batch_label = next(iter(train_loader))

    print(f"    Batch Feature Shape: {batch_feat.shape}")
    print(f"    Batch Label Shape: {batch_label.shape}")

    assert batch_feat.shape[0] == Config.BATCH_SIZE
    assert batch_feat.shape[1] == 1  # Channel dim
    assert batch_label.shape[0] == Config.BATCH_SIZE
    print("    Dataset/DataLoader verification passed.")

    # ---------------------------------------------------------
    # 5. Verify Model
    # ---------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")
    model = ConvNeXtAudio(num_classes=Config.NUM_CLASSES, pretrained=False)

    # Forward pass with the batch from previous step
    outputs = model(batch_feat)

    print(f"    Model Output Shape: {outputs.shape}")

    # Validation
    # Expected: (Batch, NumClasses)
    assert outputs.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 6. Verify Trainer (Fit and Predict)
    # ---------------------------------------------------------
    print("\n[6] Verifying Trainer Pipeline (Fit & Predict)...")
    trainer = Trainer()

    # Run Training
    print("    Starting training loop (1 epoch)...")
    trainer.fit(debug=True)

    # Verify Model Saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("    Training completed and model saved.")

    # Run Inference
    print("    Starting inference on test set...")
    trainer.predict(debug=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV was not generated."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(df_sub)} rows.")
    print(f"    First 3 rows:\n{df_sub.head(3)}")

    assert "fname" in df_sub.columns
    assert "label" in df_sub.columns
    assert len(df_sub) > 0
    print("    Trainer verification passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
