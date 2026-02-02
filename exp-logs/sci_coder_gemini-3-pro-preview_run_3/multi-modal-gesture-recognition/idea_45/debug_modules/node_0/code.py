import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import provided library modules
from library import config, utils, data_loader, model, trainer


def setup_demo_environment():
    """
    Overrides default configuration to create a lightweight demo environment.
    """
    print("=== Setting up Demo Environment ===")

    # Override paths to use a separate demo directory in ./working
    config.IDEA_NAME = "demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, config.IDEA_NAME, "cache")
    config.CHECKPOINT_DIR = os.path.join(
        config.WORKING_DIR, config.IDEA_NAME, "checkpoints"
    )
    config.SUBMISSION_DIR = os.path.join(
        config.WORKING_DIR, config.IDEA_NAME, "submission"
    )

    # Create directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Override hyperparameters for speed
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 4
    config.DEBUG_SAMPLE_SIZE = 10  # Only process 10 samples for the demo
    config.PATIENCE = 2

    # Set seed for reproducibility
    trainer.set_seed(42)
    print(f"Configured demo with DEBUG_SAMPLE_SIZE={config.DEBUG_SAMPLE_SIZE}")
    print(f"Working directory: {os.path.join(config.WORKING_DIR, config.IDEA_NAME)}")
    print("-" * 30)


def test_utils():
    """
    Verifies the logic of utility functions in library/utils.py.
    """
    print("=== Testing Utils ===")

    # 1. Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = utils.levenshtein_distance(seq1, seq2)
    assert dist_eq == 0.0, f"Expected distance 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2]
    seq4 = [1, 3]
    dist_diff = utils.levenshtein_distance(seq3, seq4)
    assert dist_diff == 1.0, f"Expected distance 1 for substitution, got {dist_diff}"

    # 2. Test Decoding (RLE + Filtering)
    # Background class is 0. Min duration is 5.
    # Sequence: [BG*2, Class1*5, Class2*3 (too short), Class3*6]
    raw_preds = [0, 0] + [1] * 5 + [2] * 3 + [3] * 6
    decoded = utils.decode_predictions_to_labels(raw_preds, min_length=5)

    expected = [1, 3]
    assert decoded == expected, f"Decoding failed. Expected {expected}, got {decoded}"

    print("Utils logic verified successfully.")
    print("-" * 30)


def test_data_loader():
    """
    Verifies the Data Loader pipeline: caching, loading, and shape consistency.
    """
    print("=== Testing Data Loader ===")

    # Initialize Dataset (Train split)
    # This will trigger processing from scratch because we changed the cache dir
    ds_train = data_loader.GestureDataset(
        split="train",
        load_cached_data=True,
        augment=True,
        debug_sample_size=config.DEBUG_SAMPLE_SIZE,
    )

    print(f"Train Dataset Length (Windows): {len(ds_train)}")

    if len(ds_train) > 0:
        # Check __getitem__ output
        features, targets = ds_train[0]

        # Expected Feature Shape: (WindowSize, 193) -> 180 (Skel) + 13 (Audio)
        expected_feat_shape = (config.WINDOW_SIZE, 193)
        assert (
            features.shape == expected_feat_shape
        ), f"Feature shape mismatch. Expected {expected_feat_shape}, got {features.shape}"

        # Expected Target Shape: (WindowSize,)
        expected_target_shape = (config.WINDOW_SIZE,)
        assert (
            targets.shape == expected_target_shape
        ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

        print("Data Loader shapes verified.")
    else:
        print("Warning: Dataset is empty. Check input data availability.")

    print("-" * 30)


def test_model():
    """
    Verifies the SHPAMCN model architecture and forward pass.
    """
    print("=== Testing Model Architecture ===")

    net = model.SHPAMCN()

    # Create dummy input: (Batch=2, Time=64, Features=193)
    dummy_input = torch.randn(2, 64, 193)

    # Forward pass
    outputs = net(dummy_input)

    # Check Deep Supervision output structure
    assert isinstance(outputs, list), "Model output should be a list (Deep Supervision)"
    assert len(outputs) == 3, "Model should return outputs for 3 stages"

    # Check output shape of the final stage: (Batch, Time, NumClasses)
    final_stage = outputs[-1]
    expected_shape = (2, 64, config.NUM_CLASSES)
    assert (
        final_stage.shape == expected_shape
    ), f"Output shape mismatch. Expected {expected_shape}, got {final_stage.shape}"

    print("Model architecture verified.")
    print("-" * 30)


def test_trainer_execution():
    """
    Runs a complete training and inference loop using the Trainer class.
    """
    print("=== Testing Trainer Execution ===")

    # Initialize Trainer
    # Uses CUDA if available, else CPU
    t = trainer.Trainer()
    print(f"Trainer initialized on device: {t.device}")

    # 1. Run Training Loop
    print("Starting Training Loop...")
    t.run_training(epochs=config.NUM_EPOCHS, debug_sample_size=config.DEBUG_SAMPLE_SIZE)

    # Verify checkpoint creation
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print("Training completed and checkpoint saved.")

    # 2. Run Prediction Loop
    print("Starting Prediction Loop...")
    t.predict_test(debug_sample_size=config.DEBUG_SAMPLE_SIZE)

    # Verify submission file creation
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Check submission content
    df_sub = pd.read_csv(submission_path, header=None)
    print(f"Submission file generated with {len(df_sub)} rows.")

    print("Trainer execution verified.")
    print("-" * 30)


if __name__ == "__main__":
    try:
        setup_demo_environment()
        test_utils()
        test_data_loader()
        test_model()
        test_trainer_execution()
        print("\nSUCCESS: All demonstrations and validations passed.")
    except AssertionError as e:
        print(f"\nFAILURE: Validation check failed -> {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred -> {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
