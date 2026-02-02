import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import library modules
import library.config
import library.utils
import library.data_loader
import library.model
import library.trainer
import library.inference


def main():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Setup & Data Preparation
    # ==========================================
    print("\n[Step 1] Preparing Mini Datasets...")

    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    # Define paths for mini metadata
    mini_train_path = os.path.join(working_dir, "mini_train.csv")
    mini_val_path = os.path.join(working_dir, "mini_val.csv")
    mini_test_path = os.path.join(working_dir, "mini_test.csv")
    demo_cache_dir = os.path.join(working_dir, "demo_cache")
    demo_model_path = os.path.join(working_dir, "demo_model.pth")
    demo_submission_path = os.path.join(working_dir, "demo_submission.csv")

    # Read original metadata and save subsets
    # We use the existing metadata files provided in the environment
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take top 4 samples for speed
    orig_train.head(4).to_csv(mini_train_path, index=False)
    orig_val.head(4).to_csv(mini_val_path, index=False)
    orig_test.head(4).to_csv(mini_test_path, index=False)

    print(f"Created mini metadata files in {working_dir}")

    # ==========================================
    # 2. Patching Configurations
    # ==========================================
    print("\n[Step 2] Patching Library Configurations for Demo...")

    # Patch library.data_loader constants
    library.data_loader.TRAIN_METADATA_PATH = mini_train_path
    library.data_loader.VAL_METADATA_PATH = mini_val_path
    library.data_loader.TEST_METADATA_PATH = mini_test_path
    library.data_loader.CACHE_DIR = demo_cache_dir
    library.data_loader.BATCH_SIZE = 2

    # Patch library.trainer constants
    library.trainer.NUM_EPOCHS = 1
    library.trainer.BATCH_SIZE = 2
    library.trainer.EARLY_STOPPING_PATIENCE = 1
    library.trainer.MODEL_SAVE_PATH = demo_model_path

    # Patch library.inference constants
    library.inference.BATCH_SIZE = 2
    library.inference.MODEL_SAVE_PATH = demo_model_path
    library.inference.SUBMISSION_FILE = demo_submission_path

    # Monkey-patch get_data_loaders in trainer and inference to force num_workers=0
    # This avoids multiprocessing overhead/issues in this short script
    original_get_loaders = library.data_loader.get_data_loaders

    def mocked_get_data_loaders(batch_size=2, num_workers=0):
        # Force num_workers=0 and use the patched paths implicitly via the module patch above
        return original_get_loaders(batch_size=batch_size, num_workers=0)

    library.trainer.get_data_loaders = mocked_get_data_loaders
    library.inference.get_data_loaders = mocked_get_data_loaders

    # ==========================================
    # 3. Demonstrate Utils
    # ==========================================
    print("\n[Step 3] Demonstrating Utils...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = library.utils.levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Expected distance 0, got {dist_eq}"

    seq3 = [1, 2, 4]
    dist_diff = library.utils.levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1, got {dist_diff}"
    print("Levenshtein distance check passed.")

    # Test Filter Short Segments
    # Sequence: 1, 1, 1, 1, 1 (5 times), 2, 2 (2 times), 0, 0, 3, 3, 3, 3, 3 (5 times)
    # Min duration is 5 by default in config.
    # Class 0 is background.
    # Expected: [1, 3] (2 is too short, 0 is background)
    raw_preds = [1] * 5 + [2] * 2 + [0] * 2 + [3] * 5
    filtered = library.utils.filter_short_segments(raw_preds, min_duration=5)
    assert filtered == [1, 3], f"Expected [1, 3], got {filtered}"
    print("Filter short segments check passed.")

    # Test LogSpaceSmoothingLoss
    criterion = library.utils.LogSpaceSmoothingLoss(threshold=1.0)
    # Create dummy log probs (Batch=1, Time=3, Classes=2)
    # Time 0: [0, -inf], Time 1: [0, -inf] -> Diff 0 -> Loss 0
    log_probs = torch.zeros(1, 3, 2)
    loss = criterion(log_probs)
    assert (
        loss.item() == 0.0
    ), f"Expected loss 0.0 for constant input, got {loss.item()}"
    print("LogSpaceSmoothingLoss check passed.")

    # ==========================================
    # 4. Demonstrate Data Loading
    # ==========================================
    print("\n[Step 4] Demonstrating Data Loader...")

    # Instantiate dataset manually to check loading
    dataset = library.data_loader.GestureDataset(
        metadata_path=mini_train_path,
        root_dir="./input",
        cache_dir=demo_cache_dir,
        is_train=True,
        load_cached=False,  # Force processing
    )

    print(f"Dataset loaded with {len(dataset)} windows.")
    assert len(dataset) > 0, "Dataset should not be empty."

    # Check item shape
    x, y, _, _ = dataset[0]
    # x shape: (Time, Features) -> (64, 193) where 193 = 20*9 + 13
    expected_features = (20 * 9) + 13
    assert (
        x.shape[1] == expected_features
    ), f"Expected {expected_features} features, got {x.shape[1]}"
    assert x.shape[0] == 64, f"Expected window size 64, got {x.shape[0]}"
    print(f"Data item shape verified: {x.shape}")

    # ==========================================
    # 5. Demonstrate Model
    # ==========================================
    print("\n[Step 5] Demonstrating Model...")

    model = library.model.RGHCMN()
    model.eval()

    # Create dummy batch (Batch=2, Time=64, Features=193)
    dummy_input = torch.randn(2, 64, expected_features)

    with torch.no_grad():
        outputs = model(dummy_input)

    # Check outputs
    assert "logits_1" in outputs
    assert "logits_2" in outputs
    assert "logits_3" in outputs

    # Logits shape should be (Batch, Time, NumClasses)
    # NumClasses = 21 (20 gestures + 1 background)
    logits_shape = outputs["logits_3"].shape
    assert logits_shape == (
        2,
        64,
        21,
    ), f"Expected logits shape (2, 64, 21), got {logits_shape}"
    print("Model forward pass successful.")

    # ==========================================
    # 6. Demonstrate Training
    # ==========================================
    print("\n[Step 6] Demonstrating Trainer...")

    trainer = library.trainer.ModelTrainer()

    # Run training (1 epoch, mini dataset)
    # This will use the patched get_data_loaders and constants
    trainer.train()

    # Verify model file was saved
    assert os.path.exists(demo_model_path), "Model file was not saved after training."
    print(f"Training completed. Model saved to {demo_model_path}")

    # ==========================================
    # 7. Demonstrate Inference
    # ==========================================
    print("\n[Step 7] Demonstrating Inference...")

    predictor = library.inference.SequencePredictor(
        model_path=demo_model_path, output_file=demo_submission_path
    )

    # Run inference
    predictor.run()

    # Verify submission file
    assert os.path.exists(demo_submission_path), "Submission file was not generated."

    # Check content
    with open(demo_submission_path, "r") as f:
        lines = f.readlines()
        print(f"Generated {len(lines)} prediction lines.")
        # We used mini_test.csv which has 4 samples
        assert len(lines) == 4, f"Expected 4 prediction lines, got {len(lines)}"
        print(f"Sample prediction: {lines[0].strip()}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    library.config.seed_everything(42)
    main()
