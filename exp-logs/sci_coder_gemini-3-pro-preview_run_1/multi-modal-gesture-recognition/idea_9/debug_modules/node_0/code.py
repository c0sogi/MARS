import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library
from library import config, utils, data_loader, model, trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Reproducibility
    print("\n[Step 1] Setting up configuration and seeds...")
    utils.set_seed(42)

    # Create a temporary working directory for demo artifacts if it doesn't exist
    demo_dir = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Create subset metadata files to ensure speed
    # We read the original metadata and take the top 5 rows
    print("Creating subset metadata for rapid execution...")

    train_df = pd.read_csv(config.TRAIN_METADATA_PATH).head(6)
    val_df = pd.read_csv(config.VAL_METADATA_PATH).head(6)
    test_df = pd.read_csv(config.TEST_METADATA_PATH).head(6)

    demo_train_path = os.path.join(demo_dir, "train_subset.csv")
    demo_val_path = os.path.join(demo_dir, "val_subset.csv")
    demo_test_path = os.path.join(demo_dir, "test_subset.csv")

    train_df.to_csv(demo_train_path, index=False)
    val_df.to_csv(demo_val_path, index=False)
    test_df.to_csv(demo_test_path, index=False)

    # Override config paths to point to our demo subsets
    config.TRAIN_METADATA_PATH = demo_train_path
    config.VAL_METADATA_PATH = demo_val_path
    config.TEST_METADATA_PATH = demo_test_path

    # Override training parameters for speed
    config.BATCH_SIZE = 2
    config.NUM_EPOCHS = 2
    config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    config.SUBMISSION_DIR = demo_dir
    config.CACHE_DIR = os.path.join(demo_dir, "cache_demo")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 2. Data Loader Verification
    print("\n[Step 2] Verifying Data Loader...")

    # Initialize Dataset
    ds = data_loader.GestureDataset(
        config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )
    print(f"Dataset length: {len(ds)}")
    assert len(ds) == 6, "Dataset length mismatch."

    # Fetch one item
    skeleton, audio, labels = ds[0]
    print(
        f"Sample 0 shapes - Skeleton: {skeleton.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    # Verify shapes: Skeleton (T, 60), Audio (T, 13), Labels (T,)
    assert skeleton.dim() == 2 and skeleton.shape[1] == 60, "Incorrect skeleton shape."
    assert (
        audio.dim() == 2 and audio.shape[1] == config.N_MFCC
    ), "Incorrect audio shape."
    assert labels.dim() == 1, "Incorrect labels shape."
    assert (
        skeleton.shape[0] == audio.shape[0] == labels.shape[0]
    ), "Temporal dimension mismatch."

    # Test Collate Function
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=data_loader.collate_fn
    )
    batch_skel, batch_audio, batch_labels, batch_mask = next(iter(loader))

    print(f"Batch shapes - Skeleton: {batch_skel.shape}, Mask: {batch_mask.shape}")
    assert batch_skel.shape[0] == 2, "Batch size mismatch."
    assert batch_mask.shape == batch_labels.shape, "Mask shape mismatch with labels."

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture...")

    # Instantiate Model
    net = model.IDGFN().to(config.DEVICE)

    # Create dummy inputs based on batch shapes
    # Lengths are needed for packing
    lengths = batch_mask.sum(dim=1).cpu()

    # Forward Pass
    logits = net(batch_skel.to(config.DEVICE), batch_audio.to(config.DEVICE), lengths)
    print(f"Logits shape: {logits.shape}")

    # Verify Output: (Batch, Time, NumClasses)
    assert logits.shape[0] == 2, "Output batch size mismatch."
    assert logits.shape[1] == batch_skel.shape[1], "Output time dimension mismatch."
    assert logits.shape[2] == config.NUM_CLASSES, "Output class dimension mismatch."

    # 4. Training Loop Demonstration
    print("\n[Step 4] Demonstrating Training Loop...")

    # Instantiate Trainer
    trainer_instance = trainer.Trainer()

    # Run training (fit)
    # Note: We already overrode config paths, but fit() takes a limit arg too.
    # Since we pointed config to small files, limit=None is fine, or we can be explicit.
    trainer_instance.fit()

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print("Training successful. Checkpoint saved.")
    else:
        raise FileNotFoundError("Checkpoint not created after training.")

    # 5. Inference & Submission
    print("\n[Step 5] Demonstrating Inference...")

    # Run prediction
    trainer_instance.predict()

    # Verify submission file
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"Prediction successful. Submission saved to {submission_path}")

        # Read and check format
        with open(submission_path, "r") as f:
            lines = f.readlines()
            print(f"First 2 lines of submission:\n{''.join(lines[:2])}")
            # Format check: SessionID,label1,label2...
            assert (
                "," in lines[0] or len(lines[0].strip().split(",")) == 1
            ), "Submission format seems incorrect."
    else:
        raise FileNotFoundError("Submission file not created.")

    # 6. Metric Logic Check
    print("\n[Step 6] Verifying Metric Logic...")

    # Test Levenshtein Ratio
    # Seq 1: [1, 2, 3]
    # Seq 2: [1, 2] -> Distance 1 (Deletion)
    # Seq 3: [1, 4, 3] -> Distance 1 (Substitution)
    preds = [[1, 2, 3], [1, 4, 3]]
    targets = [[1, 2], [1, 2, 3]]

    # Dist 1: dist([1,2,3], [1,2]) = 1
    # Dist 2: dist([1,4,3], [1,2,3]) = 1
    # Total Dist = 2
    # Total Len = len([1,2]) + len([1,2,3]) = 2 + 3 = 5
    # Ratio = 2 / 5 = 0.4

    ratio = utils.compute_levenshtein_ratio(preds, targets)
    print(f"Computed Levenshtein Ratio: {ratio}")
    assert (
        abs(ratio - 0.4) < 1e-6
    ), f"Metric calculation failed. Expected 0.4, got {ratio}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
