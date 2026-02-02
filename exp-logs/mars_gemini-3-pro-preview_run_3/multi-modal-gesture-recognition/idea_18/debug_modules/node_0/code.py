import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import provided library modules
from library import utils
from library.data_loader import GestureDataset
from library.model import AKCIRN, CascadedSmoothLoss
from library.trainer import Trainer
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Reproducibility
    utils.set_seed(42)
    print("Random seed set.")

    # 2. Prepare Mini-Dataset for Speed
    # The Trainer class expects metadata to be in {base_dir}/metadata/*.csv
    # We will create a temporary environment in ./working/demo_env
    base_demo_dir = "./working/demo_env"
    meta_demo_dir = os.path.join(base_demo_dir, "metadata")
    cache_demo_dir = os.path.join(base_demo_dir, "cache")
    submission_demo_dir = os.path.join(base_demo_dir, "submission")

    if os.path.exists(base_demo_dir):
        shutil.rmtree(base_demo_dir)
    os.makedirs(meta_demo_dir)
    os.makedirs(cache_demo_dir)
    os.makedirs(submission_demo_dir)

    print(f"Created temporary demo environment at {base_demo_dir}")

    # Read original metadata and subset top 5 samples
    # We use existing files from the read-only ./metadata directory
    for split in ["train", "val", "test"]:
        src_csv = f"./metadata/{split}.csv"
        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            # Take a small subset (5 samples) to ensure speed
            df_subset = df.head(5)
            dst_csv = os.path.join(meta_demo_dir, f"{split}.csv")
            df_subset.to_csv(dst_csv, index=False)
            print(f"Created subset metadata for {split}: {len(df_subset)} samples")

    # ==========================================
    # 3. Demonstrate library/utils.py
    # ==========================================
    print("\n--- Testing Utils ---")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = utils.compute_levenshtein(seq1, seq2)
    assert dist_eq == 0, f"Expected distance 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = utils.compute_levenshtein(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1, got {dist_diff}"
    print("Levenshtein distance check passed.")

    # Test Decoding (RLE + Background Filtering)
    # 0 is background, 1-20 are gestures
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 2, 0, 0, 1]
    # Expected: Collapse duplicates -> [0, 1, 0, 2, 0, 1] -> Filter 0 -> [1, 2, 1]
    decoded = utils.decode_predictions_to_sequence(raw_preds)
    assert decoded == [1, 2, 1], f"Expected [1, 2, 1], got {decoded}"
    print("Sequence decoding check passed.")

    # ==========================================
    # 4. Demonstrate library/data_loader.py
    # ==========================================
    print("\n--- Testing Data Loader ---")

    # Initialize Dataset with the mini metadata
    # We disable augmentation for deterministic check
    train_meta_path = os.path.join(meta_demo_dir, "train.csv")
    ds = GestureDataset(
        metadata_file=train_meta_path,
        split="train",
        window_size=64,
        stride=64,  # Non-overlapping for this test
        cache_dir=cache_demo_dir,
        load_cached=False,  # Force processing
        augment=False,
    )

    print(f"Dataset loaded with {len(ds)} windows from subset.")

    if len(ds) > 0:
        features, labels = ds[0]
        # Features: (Window, InputDim) -> (64, 193)
        # Labels: (Window,) -> (64,)
        print(f"Feature shape: {features.shape}")
        print(f"Label shape: {labels.shape}")

        assert features.shape == (64, 193), "Incorrect feature shape"
        assert labels.shape == (64,), "Incorrect label shape"
        assert isinstance(features, torch.Tensor), "Features should be a Tensor"
        assert isinstance(labels, torch.Tensor), "Labels should be a Tensor"
    else:
        print("Warning: Dataset is empty (input samples might be too short).")

    # ==========================================
    # 5. Demonstrate library/model.py
    # ==========================================
    print("\n--- Testing Model Architecture ---")

    # Constants from model.py
    INPUT_DIM = 193
    NUM_CLASSES = 21
    HIDDEN_DIM = 64

    model_instance = AKCIRN(
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM
    )
    model_instance.eval()

    # Create dummy input: (Batch, Time, Dim)
    # Note: Dataset returns (Time, Dim), DataLoader batches to (Batch, Time, Dim)
    dummy_input = torch.randn(2, 64, INPUT_DIM)

    with torch.no_grad():
        l1, l2, l3 = model_instance(dummy_input)

    # Expected output shape: (Batch, NumClasses, Time)
    expected_shape = (2, NUM_CLASSES, 64)

    print(f"Stage 1 Output: {l1.shape}")
    print(f"Stage 2 Output: {l2.shape}")
    print(f"Stage 3 Output: {l3.shape}")

    assert l1.shape == expected_shape, "Stage 1 shape mismatch"
    assert l2.shape == expected_shape, "Stage 2 shape mismatch"
    assert l3.shape == expected_shape, "Stage 3 shape mismatch"
    print("Model forward pass check passed.")

    # Test Loss
    criterion = CascadedSmoothLoss(NUM_CLASSES)
    dummy_targets = torch.randint(0, NUM_CLASSES, (2, 64))
    loss = criterion(l1, l2, l3, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"

    # ==========================================
    # 6. Demonstrate library/trainer.py
    # ==========================================
    print("\n--- Testing Trainer (Training Loop) ---")

    # Initialize Trainer with demo paths
    # We set epochs=1 and a small batch size
    trainer_instance = Trainer(
        base_dir=base_demo_dir,
        cache_dir=cache_demo_dir,
        submission_dir=submission_demo_dir,
        batch_size=2,
        epochs=1,
        patience=1,
    )

    # Run the full pipeline
    # 1. Load Data
    trainer_instance.load_data()
    assert trainer_instance.train_loader is not None

    # 2. Setup Model
    trainer_instance.setup_model()
    assert trainer_instance.model is not None

    # 3. Train (1 Epoch)
    print("Running training for 1 epoch...")
    trainer_instance.train()

    # Check if model checkpoint was saved
    best_model_path = os.path.join(cache_demo_dir, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint not found"
    print("Training complete and model saved.")

    # 4. Generate Submission (using Trainer method)
    trainer_instance.generate_submission()
    submission_file = os.path.join(submission_demo_dir, "submission.csv")
    assert os.path.exists(submission_file), "Submission file not created by Trainer"

    # Verify submission content
    with open(submission_file, "r") as f:
        lines = f.readlines()
        print(f"Trainer Submission Lines: {len(lines)}")
        # Check format of first line
        if len(lines) > 0:
            print(f"Sample line: {lines[0].strip()}")

    # ==========================================
    # 7. Demonstrate library/inference.py
    # ==========================================
    print("\n--- Testing Inference (Standalone) ---")

    # Define separate output dir for standalone inference
    inference_sub_dir = os.path.join(base_demo_dir, "inference_submission")

    # Run inference using the model trained above
    run_inference(
        base_dir=base_demo_dir,
        cache_dir=cache_demo_dir,
        model_path=best_model_path,
        submission_dir=inference_sub_dir,
        batch_size=2,
    )

    inf_sub_file = os.path.join(inference_sub_dir, "submission.csv")
    assert os.path.exists(
        inf_sub_file
    ), "Submission file not created by Inference module"

    with open(inf_sub_file, "r") as f:
        lines = f.readlines()
        print(f"Inference Submission Lines: {len(lines)}")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
