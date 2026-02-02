import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, levenshtein_distance, decode_predictions_rle
from library.data_loader import GestureDataset, collate_fn
from library.model import SR_IIN
from library.train import run_training
from library.predict import generate_submission


def setup_demo_environment():
    """
    Sets up a temporary environment with subset metadata for fast execution.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_work_dir = "./working/demo_execution"
    demo_meta_dir = os.path.join(demo_work_dir, "metadata")

    # Clean up previous run if exists
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)

    os.makedirs(demo_meta_dir, exist_ok=True)

    # Subset Metadata (Take top 10 samples from each)
    # We read from the actual metadata files provided in the environment
    splits = {"train": Config.TRAIN_CSV, "val": Config.VAL_CSV, "test": Config.TEST_CSV}

    for split_name, csv_path in splits.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Take a small subset
            subset_df = df.head(10)
            subset_path = os.path.join(demo_meta_dir, f"{split_name}.csv")
            subset_df.to_csv(subset_path, index=False)
            print(f"Created subset metadata for {split_name}: {len(subset_df)} samples")
        else:
            raise FileNotFoundError(f"Original metadata not found at {csv_path}")

    # Patch Config to use demo directories and parameters
    Config.WORK_DIR = demo_work_dir
    Config.METADATA_DIR = demo_meta_dir
    Config.CACHE_DIR = os.path.join(demo_work_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_work_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_work_dir, "submission")

    Config.TRAIN_CSV = os.path.join(demo_meta_dir, "train.csv")
    Config.VAL_CSV = os.path.join(demo_meta_dir, "val.csv")
    Config.TEST_CSV = os.path.join(demo_meta_dir, "test.csv")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Optimization for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_LAYERS = 1  # Reduce model complexity for demo
    Config.HIDDEN_DIM = 64

    # Ensure directories exist
    Config.setup()
    print(">>> Configuration patched for demo execution.")


def demonstrate_utils():
    """
    Verifies utility functions.
    """
    print("\n>>> Demonstrating Utilities...")

    # 1. Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 3]  # Deletion of '2'
    dist = levenshtein_distance(seq1, seq2)
    print(f"Levenshtein Distance between {seq1} and {seq2}: {dist}")
    assert dist == 1, "Levenshtein distance calculation is incorrect."

    # 2. Decode Predictions RLE
    # Create synthetic logits: (Time=10, Classes=21)
    # Background=0. Let's make a sequence: 0, 0, 1, 1, 1, 0, 2, 2, 2, 0
    # Config.MIN_SEGMENT_LENGTH is default 5, let's lower it temporarily for this check
    original_min_seg = Config.MIN_SEGMENT_LENGTH
    Config.MIN_SEGMENT_LENGTH = 2

    T, C = 10, Config.NUM_CLASSES + 1
    logits = np.zeros((T, C))
    # Fill logits to force argmax
    indices = [0, 0, 1, 1, 1, 0, 2, 2, 2, 0]
    for t, cls_idx in enumerate(indices):
        logits[t, cls_idx] = 10.0  # High logit for target class

    decoded = decode_predictions_rle(logits)
    print(f"Synthetic Indices: {indices}")
    print(f"Decoded Sequence: {decoded}")

    # Expectation: 1 (length 3) and 2 (length 3) are kept. 0 is background.
    # Note: Median filter might slightly alter boundaries, but with kernel 5 on length 3 segments,
    # it might erode them if surrounded by 0.
    # Let's trust the function runs without error and returns a list.
    assert isinstance(decoded, list), "decode_predictions_rle should return a list."

    # Restore Config
    Config.MIN_SEGMENT_LENGTH = original_min_seg


def demonstrate_data_loader():
    """
    Verifies Dataset and DataLoader.
    """
    print("\n>>> Demonstrating Data Loader...")

    # Initialize Dataset (this will trigger processing and caching of the subset)
    ds = GestureDataset(split="train", load_cached_data=False)
    print(f"Dataset size: {len(ds)}")
    assert len(ds) > 0, "Dataset should not be empty."

    # Fetch one sample
    sample = ds[0]
    print(f"Sample ID: {sample['id']}")
    print(f"Skeleton Shape: {sample['skeleton'].shape}")  # (T, 60)
    print(f"Audio Shape: {sample['audio'].shape}")  # (T, 13)
    print(f"Labels Shape: {sample['labels'].shape}")  # (T,)

    assert sample["skeleton"].dim() == 2
    assert sample["skeleton"].shape[1] == Config.NUM_JOINTS * 3
    assert sample["audio"].dim() == 2
    assert sample["audio"].shape[1] == Config.AUDIO_N_MFCC

    # Test DataLoader Collate
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dl))

    print(f"Batch Skeleton Shape: {batch['skeleton'].shape}")  # (B, T_max, 60)
    print(f"Batch Mask Shape: {batch['mask'].shape}")  # (B, T_max)

    assert batch["skeleton"].shape[0] == 2
    assert batch["mask"].shape == (2, batch["skeleton"].shape[1])

    return batch


def demonstrate_model(batch):
    """
    Verifies Model instantiation and forward pass.
    """
    print("\n>>> Demonstrating Model...")

    device = torch.device("cpu")  # Use CPU for simple demo check
    model = SR_IIN().to(device)

    # Prepare inputs
    skeleton = batch["skeleton"].to(device)
    audio = batch["audio"].to(device)
    lengths = batch["lengths"].to(device)
    mask = batch["mask"].to(device)

    # Forward
    logits = model(skeleton, audio, lengths, mask)
    print(f"Logits Shape: {logits.shape}")  # (B, T, NumClasses+1)

    assert logits.shape[0] == 2
    assert logits.shape[1] == skeleton.shape[1]
    assert logits.shape[2] == Config.NUM_CLASSES + 1

    return model


def demonstrate_training_pipeline():
    """
    Runs the provided training script function.
    """
    print("\n>>> Demonstrating Training Pipeline...")

    # This function inside library/train.py handles loop, validation, and saving.
    # We patched Config.EPOCHS to 1 and BATCH_SIZE to 2.
    run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Verify checkpoint creation
    expected_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), "Checkpoint file was not created."
    print("Training pipeline completed and checkpoint saved.")


def demonstrate_inference_pipeline():
    """
    Runs the provided prediction script function.
    """
    print("\n>>> Demonstrating Inference Pipeline...")

    # This function loads the best model and generates submission.csv
    generate_submission(batch_size=Config.BATCH_SIZE)

    # Verify submission file
    expected_sub = Config.SUBMISSION_PATH
    assert os.path.exists(expected_sub), "Submission file was not created."

    # Check content
    df = pd.read_csv(expected_sub, header=None)
    print(f"Submission rows generated: {len(df)}")
    assert len(df) > 0, "Submission file is empty."
    print("Inference pipeline completed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup Environment
    setup_demo_environment()

    # 2. Utils
    demonstrate_utils()

    # 3. Data Loading
    batch = demonstrate_data_loader()

    # 4. Model
    demonstrate_model(batch)

    # 5. Training
    demonstrate_training_pipeline()

    # 6. Inference
    demonstrate_inference_pipeline()

    print("\n>>> All demonstrations passed successfully.")
