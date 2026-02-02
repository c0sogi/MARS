import sys
import os
import torch
import numpy as np

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import GestureDataset, collate_fn
from library.model import MSRN
from library.losses import HierarchicalLoss
from library.trainer import Trainer
from library.inference import InferenceManager


def main():
    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    print(">>> 1. Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use main thread for simple demo to avoid overhead

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Cache Directory: {Config.CACHE_DIR}")

    # ==========================================
    # 2. Data Loader Verification
    # ==========================================
    print("\n>>> 2. Verifying Data Loader and Dataset...")

    # Limit to a small number of samples for speed
    limit_samples = 8

    # Instantiate Training Dataset
    print(f"    Initializing GestureDataset (limit={limit_samples})...")
    train_dataset = GestureDataset(split="train", augment=True, limit=limit_samples)

    # Verify dataset length
    assert (
        len(train_dataset) == limit_samples
    ), f"Dataset length mismatch. Expected {limit_samples}, got {len(train_dataset)}"

    # Fetch a single item
    skeleton, audio, labels = train_dataset[0]
    print(
        f"    Sample 0 Shapes -> Skeleton: {skeleton.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    # Verify Item Shapes
    # Skeleton: (Time, Joints * 3)
    assert (
        skeleton.dim() == 2 and skeleton.shape[1] == Config.INPUT_DIM_SKELETON
    ), f"Skeleton shape incorrect: {skeleton.shape}"
    # Audio: (Time, MFCC)
    assert (
        audio.dim() == 2 and audio.shape[1] == Config.INPUT_DIM_AUDIO
    ), f"Audio shape incorrect: {audio.shape}"
    # Labels: (Time,)
    assert labels.dim() == 1, f"Labels shape incorrect: {labels.shape}"

    # Verify Collate Function
    print("    Testing collate_fn with a batch...")
    batch = [train_dataset[i] for i in range(Config.BATCH_SIZE)]
    padded_skel, padded_audio, padded_labels, lengths = collate_fn(batch)

    print(
        f"    Batch Shapes -> Skel: {padded_skel.shape}, Audio: {padded_audio.shape}, Labels: {padded_labels.shape}"
    )

    assert padded_skel.shape[0] == Config.BATCH_SIZE
    assert padded_audio.shape[0] == Config.BATCH_SIZE
    assert padded_labels.shape[0] == Config.BATCH_SIZE
    # Check padding logic (sequences should be padded to max length in batch)
    assert padded_skel.shape[1] == lengths.max().item()

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n>>> 3. Verifying Model Architecture...")

    model = MSRN().to(device)

    # Move batch to device
    skel_in = padded_skel.to(device)
    audio_in = padded_audio.to(device)

    # Forward Pass
    stage1_logits, stage2_logits = model(skel_in, audio_in)

    print(
        f"    Output Shapes -> Stage1: {stage1_logits.shape}, Stage2: {stage2_logits.shape}"
    )

    # Verify Output Dimensions: (Batch, Time, NumClasses)
    expected_shape = (Config.BATCH_SIZE, padded_skel.shape[1], Config.NUM_CLASSES)
    assert stage1_logits.shape == expected_shape, "Stage 1 logits shape mismatch"
    assert stage2_logits.shape == expected_shape, "Stage 2 logits shape mismatch"

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n>>> 4. Verifying Loss Calculation...")

    criterion = HierarchicalLoss(ignore_index=-100).to(device)
    labels_in = padded_labels.to(device)

    loss = criterion(stage1_logits, stage2_logits, labels_in)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss must be positive"

    # ==========================================
    # 5. Training Loop Verification
    # ==========================================
    print("\n>>> 5. Verifying Trainer (Training Loop)...")

    # Initialize Trainer with limited data
    trainer = Trainer(limit=limit_samples)

    # Run fit (Config.NUM_EPOCHS is set to 1)
    print("    Starting training epoch...")
    trainer.fit(epochs=Config.NUM_EPOCHS)

    # Verify Model Checkpoint creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print("    Training complete and model saved.")

    # ==========================================
    # 6. Inference and Submission Verification
    # ==========================================
    print("\n>>> 6. Verifying Inference Pipeline...")

    inference_manager = InferenceManager()

    # Run prediction on a small subset of test data
    print("    Running inference on test subset...")
    inference_manager.predict_all(limit=limit_samples)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Validate content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        print(f"    Generated {len(lines)} submission lines.")

        if len(lines) > 0:
            sample_line = lines[0].strip()
            print(f"    Sample Output: {sample_line}")
            parts = sample_line.split(",")

            # Check ID format
            assert (
                "Sample" in parts[0] or "Session" in parts[0]
            ), "Invalid Sample ID in submission"

            # Check label format (if gestures were detected)
            if len(parts) > 1:
                assert all(
                    p.isdigit() for p in parts[1:]
                ), "Gesture labels must be integers"

    print("\n>>> All verification steps passed successfully!")


if __name__ == "__main__":
    main()
