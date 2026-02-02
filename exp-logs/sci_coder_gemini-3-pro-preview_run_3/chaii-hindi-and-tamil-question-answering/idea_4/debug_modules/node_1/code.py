import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed, jaccard, decode_bio_spans
from library.tapt_engine import run_tapt
from library.qa_data import prepare_qa_data
from library.qa_trainer import train_fold
from library.inference_engine import predict_and_submit


def setup_demo_environment():
    """
    Sets up a temporary directory for the demo and overrides Config parameters
    to ensure the script runs quickly and does not interfere with existing work.
    """
    print(">>> Setting up demo environment...")

    # Define demo working directory
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create a metadata subdirectory for subsampled data
    demo_meta_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Override Config paths to point to the demo directory
    Config.WORKING_DIR = demo_dir
    Config.METADATA_DIR = demo_meta_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Update derived paths manually since they were initialized at import time
    Config.TAPT_CACHE_DIR = os.path.join(demo_dir, "tapt_cache")
    Config.QA_CACHE_DIR = os.path.join(demo_dir, "qa_cache")
    Config.TAPT_MODEL_DIR = os.path.join(demo_dir, "tapt_model_finetuned")
    Config.QA_MODELS_DIR = os.path.join(demo_dir, "qa_models")

    # Override Metadata paths to point to our new small files
    Config.TRAIN_META_PATH = os.path.join(demo_meta_dir, "train.csv")
    Config.VAL_META_PATH = os.path.join(demo_meta_dir, "val.csv")
    Config.TEST_META_PATH = os.path.join(demo_meta_dir, "test.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.TAPT_EPOCHS = 1
    Config.SEEDS = [42]  # Run only one seed for demo
    Config.BATCH_SIZE = 4
    Config.TAPT_BATCH_SIZE = 4

    # Create directories
    Config.setup()

    return demo_meta_dir


def create_subsampled_data(original_meta_dir, target_meta_dir, sample_size=20):
    """
    Reads the original metadata, samples a few rows, and saves them to the
    demo metadata directory.
    """
    print(f">>> Creating subsampled datasets (size={sample_size})...")

    files = ["train.csv", "val.csv", "test.csv"]
    for fname in files:
        src_path = os.path.join(original_meta_dir, fname)
        dst_path = os.path.join(target_meta_dir, fname)

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Sample to reduce runtime
            df_small = df.head(sample_size).copy()
            df_small.to_csv(dst_path, index=False)
            print(f"    Created {dst_path} with {len(df_small)} rows.")
        else:
            print(f"    Warning: Source file {src_path} not found.")


def verify_utils():
    """
    Validates utility functions.
    """
    print(">>> Verifying utility functions...")

    # Test Jaccard
    s1 = "hello world"
    s2 = "hello python world"
    score = jaccard(s1, s2)
    # intersection: hello, world (2). union: hello, world, python (3). score: 2/3
    assert abs(score - 0.6666) < 0.001, f"Jaccard calculation failed: {score}"

    # Test BIO Decoding
    context = "The answer is 42."
    # Tokens (approx): [The, answer, is, 42, .]
    # Offsets (dummy): [(0,3), (4,10), (11,13), (14,16), (16,17)]
    offsets = [(0, 3), (4, 10), (11, 13), (14, 16), (16, 17)]
    tags = [0, 0, 0, 1, 0]  # "42" is B-ANS
    decoded = decode_bio_spans(context, tags, offsets)
    assert decoded == "42", f"BIO decoding failed. Got: '{decoded}'"

    print("    Utils verification passed.")


def main():
    # 1. Setup
    set_seed(42)
    original_meta_dir = "./metadata"
    demo_meta_dir = setup_demo_environment()

    # 2. Prepare Data
    create_subsampled_data(original_meta_dir, demo_meta_dir, sample_size=10)

    # 3. Verify Utils
    verify_utils()

    # 4. Run TAPT (Task-Adaptive Pretraining)
    print("\n>>> Step 4: Running TAPT...")
    # This will generate a corpus from our small metadata and fine-tune the model for 1 epoch
    run_tapt()

    # Verify TAPT output
    assert os.path.exists(Config.TAPT_MODEL_DIR), "TAPT model directory not created."
    # Check for model weights (safetensors or bin)
    has_weights = any(
        f.endswith(".bin") or f.endswith(".safetensors")
        for f in os.listdir(Config.TAPT_MODEL_DIR)
    )
    assert has_weights, "TAPT model weights not found."
    print("    TAPT completed successfully.")

    # 5. Prepare QA Data
    print("\n>>> Step 5: Preparing QA Data...")
    # load_cached_data=False forces processing from our new small CSVs
    train_dataset, val_dataset, test_dataset, test_features = prepare_qa_data(
        load_cached_data=False
    )

    assert len(train_dataset) > 0, "Train dataset is empty."
    assert len(test_features) > 0, "Test features are empty."
    print(f"    QA Data prepared. Train size: {len(train_dataset)}")

    # 6. Train QA Model
    print("\n>>> Step 6: Training QA Model...")
    # Train using the first seed defined in Config.SEEDS (we set it to [42])
    train_fold(train_dataset, val_dataset, seed=42)

    # Verify QA Model output
    model_path = os.path.join(Config.QA_MODELS_DIR, "model_seed_42.pt")
    assert os.path.exists(model_path), f"QA model file not found at {model_path}"
    print("    QA Training completed successfully.")

    # 7. Inference and Submission
    print("\n>>> Step 7: Running Inference...")
    predict_and_submit(test_dataset, test_features)

    # Verify Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found."

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission generated with {len(df_sub)} rows.")
    print("    Head of submission:")
    print(df_sub.head())

    # Check format
    assert (
        "id" in df_sub.columns and "PredictionString" in df_sub.columns
    ), "Invalid submission columns."

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    main()
