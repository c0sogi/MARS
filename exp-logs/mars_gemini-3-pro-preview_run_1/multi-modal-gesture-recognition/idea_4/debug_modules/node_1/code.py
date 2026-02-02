import sys
import os
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_levenshtein, decode_predictions
from library.data_loader import GestureDataset, CollateFn
from library.model import CGR_GRU
from library.train import Trainer


def run_demo():
    print("=== Starting CGR-GRU Demo Script ===")

    # 1. Configuration Overrides for Speed and Isolation
    print("\n[1] Configuring environment...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache_train")
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Create working directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Hyperparameters for fast execution
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 2  # Small batch size
    Config.HIDDEN_DIM = 64  # Smaller model dimension

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("    Configuration updated: Subset=10, Epochs=2, Batch=2.")

    # 2. Verify Utilities
    print("\n[2] Verifying Utilities...")

    # Test Levenshtein
    seq_a = [1, 2, 3]
    seq_b = [1, 3]  # Deletion of '2'
    dist = compute_levenshtein(seq_a, seq_b)
    assert dist == 1.0, f"Levenshtein calculation incorrect. Expected 1.0, got {dist}"
    print("    Levenshtein distance check passed.")

    # Test Decode Predictions
    # Create synthetic probabilities for 20 frames
    # Frames 0-9: Class 1, Frames 10-19: Class 2
    T_frames = 20
    probs = np.zeros((T_frames, Config.NUM_CLASSES))
    probs[0:10, 1] = 10.0  # High logit for class 1
    probs[10:20, 2] = 10.0  # High logit for class 2

    # decode_predictions uses a median filter (k=5) and requires segments >= 5 frames
    decoded = decode_predictions(probs)
    assert decoded == [
        1,
        2,
    ], f"Decoding logic incorrect. Expected [1, 2], got {decoded}"
    print("    Prediction decoding check passed.")

    # 3. Verify Data Loader
    print("\n[3] Verifying Data Loader...")

    # Initialize Dataset (Train)
    # This will also trigger stats computation/saving in the new WORKING_DIR
    dataset = GestureDataset(split="train", debug=True)
    print(f"    Dataset initialized with {len(dataset)} samples.")
    assert len(dataset) <= Config.DEBUG_SUBSET_SIZE, "Dataset subset size limit failed."

    # Fetch a valid sample
    sample = None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            break

    assert (
        sample is not None
    ), "Failed to load any valid samples from the dataset subset."

    skel = sample["skeleton"]
    audio = sample["audio"]
    labels = sample["labels"]

    print(
        f"    Sample Shapes -> Skeleton: {skel.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    # Verify shapes
    # Skeleton: (Time, 60)
    assert (
        skel.ndim == 2 and skel.shape[1] == 60
    ), f"Skeleton shape mismatch: {skel.shape}"
    # Audio: (Time, 13)
    assert (
        audio.ndim == 2 and audio.shape[1] == 13
    ), f"Audio shape mismatch: {audio.shape}"
    # Labels: (Time,)
    assert labels.ndim == 1, f"Labels shape mismatch: {labels.shape}"
    # Temporal sync
    assert (
        skel.shape[0] == audio.shape[0] == labels.shape[0]
    ), "Temporal dimension mismatch between modalities."

    # Test Collate Function
    collate = CollateFn(mode="train")
    batch_list = [sample, sample]  # Create a batch of 2 identical samples
    batch = collate(batch_list)

    assert batch["skeleton"].shape[0] == 2, "Batch size mismatch in collate."
    assert batch["skeleton"].shape[2] == 60, "Feature dimension mismatch in collate."
    assert "lengths" in batch, "Lengths key missing in batch."
    print("    Collate function check passed.")

    # 4. Verify Model
    print("\n[4] Verifying Model Architecture...")
    device = Config.get_device()
    model = CGR_GRU().to(device)

    # Create dummy batch
    B_test = 2
    T_test = 50
    dummy_skel = torch.randn(B_test, T_test, 60).to(device)
    dummy_audio = torch.randn(B_test, T_test, 13).to(device)
    dummy_lengths = torch.tensor([T_test, T_test], dtype=torch.long).to(device)

    # Forward pass
    logits = model(dummy_skel, dummy_audio, lengths=dummy_lengths)

    print(f"    Logits Shape: {logits.shape}")
    expected_shape = (B_test, T_test, Config.NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {logits.shape}"
    print("    Model forward pass check passed.")

    # 5. Verify Training Loop
    print("\n[5] Verifying Training Loop...")

    # Initialize Trainer
    # Note: Trainer will re-initialize datasets, picking up the Config overrides
    trainer = Trainer()

    # Run training
    print("    Starting fit()...")
    trainer.fit()

    # Check for checkpoint
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"    Checkpoint successfully saved at: {Config.CHECKPOINT_PATH}")
    else:
        # It's possible no improvement occurred in 2 epochs, but with random init usually one saves.
        # We enforce a save check failure only if we expect it to strictly succeed.
        # For demo purposes, we note it.
        print(
            "    Notice: No checkpoint saved (Validation LER might not have improved)."
        )

    # 6. Verify Inference with Trained Model
    print("\n[6] Verifying Inference...")

    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        print("    Loaded best model weights.")

    model.eval()
    with torch.no_grad():
        # Use the batch created in step 3
        skel_b = batch["skeleton"].to(device)
        audio_b = batch["audio"].to(device)
        lens_b = batch["lengths"]  # Usually CPU is fine for logic, but model handles it

        logits = model(skel_b, audio_b, lengths=lens_b)
        probs = torch.softmax(logits, dim=2)

        # Decode the first sequence in the batch
        # Slice by actual length to avoid padding effects
        seq_len = lens_b[0]
        pred_seq = decode_predictions(probs[0, :seq_len, :])
        print(f"    Inference successful. Predicted sequence: {pred_seq}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
