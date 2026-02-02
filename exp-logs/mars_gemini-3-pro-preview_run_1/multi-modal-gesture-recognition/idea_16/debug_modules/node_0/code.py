import os
import sys
import torch
import numpy as np
import warnings

# Import from the provided library files
from library.config import (
    WORK_DIR,
    CHECKPOINT_DIR,
    DEVICE,
    SKELETON_JOINTS,
    SKELETON_CHANNELS,
    AUDIO_N_MELS,
    NUM_CLASSES,
    LABEL_MAP,
)
from library.utils import (
    set_seed,
    decode_predictions,
    compute_levenshtein,
    compute_dataset_metric,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import PCA_IIN
from library.train import train_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_utilities():
    print(">>> Testing Utilities...")

    # 1. Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = compute_levenshtein(seq1, seq2)
    assert dist_eq == 0, f"Expected distance 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = compute_levenshtein(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1 for deletion, got {dist_diff}"

    # 2. Test Decode Predictions (Median Filter + RLE)
    # Create a sequence: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    # 0 is background (should be filtered), 1 and 2 are gestures.
    # Length 5 is the threshold in decode_predictions.
    raw_preds = np.array([1] * 5 + [2] * 5 + [0] * 5)
    decoded = decode_predictions(raw_preds)

    # Expect [1, 2] because 0 is background
    assert decoded == [1, 2], f"Expected [1, 2], got {decoded}"

    # Test short segment filtering (length < 5)
    # [3, 3, 3] is length 3, should be filtered out
    raw_preds_noise = np.array([1] * 6 + [3] * 3 + [2] * 6)
    decoded_noise = decode_predictions(raw_preds_noise)
    assert decoded_noise == [
        1,
        2,
    ], f"Expected [1, 2] (filtering noise), got {decoded_noise}"

    print("Utilities verified successfully.\n")


def test_data_loader():
    print(">>> Testing Data Loader...")

    # Use a small subset for debugging
    debug_size = 10

    # Instantiate Dataset (Train)
    # This will trigger stat computation if not present, but on 10 samples it's fast.
    dataset = GestureDataset(
        split="train", load_cached_data=True, debug_subset_size=debug_size
    )

    assert (
        len(dataset) <= debug_size
    ), f"Dataset size {len(dataset)} exceeds debug limit {debug_size}"

    # Fetch one sample
    skel, audio, labels = dataset[0]

    # Check dimensions
    # Skel: (Time, 60)
    assert (
        skel.ndim == 2 and skel.shape[1] == SKELETON_JOINTS * SKELETON_CHANNELS
    ), f"Skeleton shape mismatch: {skel.shape}"

    # Audio: (Time, 64)
    assert (
        audio.ndim == 2 and audio.shape[1] == AUDIO_N_MELS
    ), f"Audio shape mismatch: {audio.shape}"

    # Labels: (Time,)
    assert labels.ndim == 1, f"Labels shape mismatch: {labels.shape}"

    # Check temporal alignment (approximate due to different sampling rates, but loader aligns them)
    # The loader trims to min length
    assert (
        skel.shape[0] == audio.shape[0] == labels.shape[0]
    ), f"Temporal misalignment: Skel {skel.shape[0]}, Audio {audio.shape[0]}, Labels {labels.shape[0]}"

    # Test Collate Function
    batch_size = 4
    batch_samples = [dataset[i] for i in range(batch_size)]
    skels_pad, audios_pad, labels_pad, lengths = collate_fn(batch_samples)

    assert skels_pad.shape[0] == batch_size
    assert audios_pad.shape[0] == batch_size
    assert labels_pad.shape[0] == batch_size
    assert lengths.shape[0] == batch_size

    # Verify sorting (descending length)
    sorted_lengths, _ = torch.sort(lengths, descending=True)
    assert torch.equal(
        lengths, sorted_lengths
    ), "Batch is not sorted by length descending"

    print("Data Loader verified successfully.\n")
    return skels_pad, audios_pad, lengths


def test_model(batch_data):
    print(">>> Testing Model Architecture...")

    skels, audios, lengths = batch_data

    # Move to device
    skels = skels.to(DEVICE)
    audios = audios.to(DEVICE)
    lengths = lengths.to(DEVICE)

    model = PCA_IIN().to(DEVICE)
    model.eval()

    with torch.no_grad():
        logits = model(skels, audios, lengths)

    # Output shape: (Batch, MaxTime, NumClasses)
    assert logits.ndim == 3
    assert logits.shape[0] == skels.shape[0]
    assert logits.shape[1] == skels.shape[1]
    assert logits.shape[2] == NUM_CLASSES

    print(f"Model forward pass successful. Output shape: {logits.shape}")
    print("Model verified successfully.\n")


def test_training_loop():
    print(">>> Testing Full Training Loop...")

    # Run training for 2 epochs on a tiny subset
    # This verifies the integration of dataset, model, loss, optimizer, and validation
    debug_size = 10
    epochs = 2

    best_error_rate = train_model(debug_subset_size=debug_size, epochs=epochs)

    # Check if model checkpoint was created
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file 'best_model.pth' was not created."

    assert isinstance(best_error_rate, float), "Error rate should be a float."
    print(f"Training loop finished with Best Error Rate: {best_error_rate:.4f}")
    print("Training Loop verified successfully.\n")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    print("=== Starting Demonstration ===\n")

    # 1. Utilities
    test_utilities()

    # 2. Data Loader
    batch_data = test_data_loader()

    # 3. Model
    test_model(batch_data)

    # 4. Training Loop
    test_training_loop()

    print("=== Demonstration Complete ===")
