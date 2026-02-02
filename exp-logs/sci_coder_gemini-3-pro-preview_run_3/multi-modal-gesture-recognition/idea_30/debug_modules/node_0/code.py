import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path so we can import the library modules
sys.path.append(os.getcwd())

# Import library modules
from library import config, utils, data_loader, model, trainer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def demo_utils():
    print("\n=== Demonstrating Utils ===")

    # 1. Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 4]
    dist = utils.levenshtein_distance(seq1, seq2)
    print(f"Levenshtein distance between {seq1} and {seq2}: {dist}")
    assert dist == 1, "Levenshtein distance calculation is incorrect."

    seq3 = [1, 2]
    seq4 = [1, 2, 3, 4]
    dist2 = utils.levenshtein_distance(seq3, seq4)
    print(f"Levenshtein distance between {seq3} and {seq4}: {dist2}")
    assert dist2 == 2, "Levenshtein distance calculation is incorrect."

    # 2. Test Process Predictions (Post-processing)
    # Simulate probabilities for 10 frames, 3 classes.
    # Frames 0-4: Class 1, Frames 5-9: Class 2
    # We create a probability matrix where argmax gives the expected classes
    probs = np.zeros((10, 3))
    probs[0:5, 1] = 1.0  # Class 1
    probs[5:10, 2] = 1.0  # Class 2

    # Set min_length to 3 to keep these segments
    pred_seq = utils.process_predictions(probs, min_length=3, bg_class_idx=0)
    print(f"Processed prediction sequence: {pred_seq}")
    assert pred_seq == [1, 2], f"Expected [1, 2], got {pred_seq}"

    # Test filtering short segments
    # Frames 0-1: Class 1 (length 2), Frames 2-9: Class 2 (length 8)
    # With min_length=3, Class 1 should be filtered out
    probs_short = np.zeros((10, 3))
    probs_short[0:2, 1] = 1.0
    probs_short[2:10, 2] = 1.0

    pred_seq_short = utils.process_predictions(
        probs_short, min_length=3, bg_class_idx=0
    )
    print(f"Processed prediction sequence (filtering short): {pred_seq_short}")
    assert pred_seq_short == [2], f"Expected [2], got {pred_seq_short}"

    print("Utils demonstration passed.")


def demo_data_loader():
    print("\n=== Demonstrating Data Loader ===")

    # Override config for speed
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 5  # Only load 5 samples
    config.BATCH_SIZE = 2
    config.CACHE_DIR = "./working/demo_cache"

    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print(
        f"Loading dataloaders with DEBUG={config.DEBUG}, Batch Size={config.BATCH_SIZE}..."
    )
    train_loader, val_loader, test_loader, _, _ = data_loader.get_dataloaders(
        load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch
    features, targets = next(iter(train_loader))

    print(f"Feature shape: {features.shape}")  # Expected: (Batch, Window, InputDim)
    print(f"Target shape: {targets.shape}")  # Expected: (Batch, Window)

    # Assertions
    assert features.dim() == 3, "Features should be 3-dimensional (Batch, Time, Feat)"
    assert (
        features.shape[2] == config.INPUT_DIM
    ), f"Feature dimension mismatch. Expected {config.INPUT_DIM}, got {features.shape[2]}"
    assert targets.dim() == 2, "Targets should be 2-dimensional (Batch, Time)"
    assert (
        features.shape[1] == config.WINDOW_SIZE
    ), f"Window size mismatch. Expected {config.WINDOW_SIZE}, got {features.shape[1]}"

    print("Data Loader demonstration passed.")
    return features, targets


def demo_model(sample_features):
    print("\n=== Demonstrating Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.HSGKN().to(device)

    input_tensor = sample_features.to(device)

    print("Performing forward pass...")
    outputs = net(input_tensor)

    # Verify output structure
    assert isinstance(outputs, dict), "Model output should be a dictionary"
    assert "stage1" in outputs, "Output missing stage1"
    assert "stage2" in outputs, "Output missing stage2"
    assert "stage3" in outputs, "Output missing stage3"

    # Verify shapes
    # Output logits shape: (Batch, Classes, Time)
    s3_logits = outputs["stage3"]
    print(f"Stage 3 Logits Shape: {s3_logits.shape}")

    batch_size = input_tensor.shape[0]
    time_steps = input_tensor.shape[1]

    assert s3_logits.shape[0] == batch_size, "Batch size mismatch in output"
    assert (
        s3_logits.shape[1] == config.NUM_CLASSES
    ), f"Class count mismatch. Expected {config.NUM_CLASSES}, got {s3_logits.shape[1]}"
    assert s3_logits.shape[2] == time_steps, "Time dimension mismatch in output"

    print("Model demonstration passed.")


def demo_training_pipeline():
    print("\n=== Demonstrating Training Pipeline ===")

    # Setup paths for demo
    config.BEST_MODEL_PATH = "./working/demo_best_model.pth"
    config.SUBMISSION_PATH = "./working/demo_submission.csv"
    config.NUM_EPOCHS = 1  # Run only 1 epoch

    print(f"Initializing Trainer (Epochs={config.NUM_EPOCHS})...")
    # Trainer initializes its own loaders, but since we modified config globally,
    # it will pick up the DEBUG settings and paths.
    trainer_instance = trainer.Trainer()

    print("Starting fit()...")
    trainer_instance.fit()

    print("Checking for saved model...")
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"Model checkpoint found at {config.BEST_MODEL_PATH}")
    else:
        # It's possible validation score didn't improve if initialized with inf,
        # but the trainer logic saves if val_score < best_score (inf).
        # However, if validation fails or dataset is empty, it might not save.
        # Given our setup, it should save.
        print(
            "Warning: Model checkpoint not found. (This might happen if validation set is empty in debug mode)"
        )

    print("Starting predict()...")
    trainer_instance.predict()

    print("Checking for submission file...")
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Submission file generated at {config.SUBMISSION_PATH}")
        # Verify content
        with open(config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"Submission file has {len(lines)} lines.")
            if len(lines) > 0:
                print(f"First line: {lines[0].strip()}")
    else:
        raise AssertionError("Submission file was not generated.")

    print("Training pipeline demonstration passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)

    # 2. Utils Demo
    demo_utils()

    # 3. Data Loader Demo
    # We keep the features from this step to pass to the model demo
    sample_features, sample_targets = demo_data_loader()

    # 4. Model Demo
    demo_model(sample_features)

    # 5. Full Training/Inference Pipeline Demo
    demo_training_pipeline()

    print("\nAll demonstrations completed successfully.")
