import os
import shutil
import pandas as pd
import numpy as np
import torch
import logging

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.image_processing import ImageProcessor
from library.feature_extraction import DualStreamExtractor
from library.trainer import train_ensemble, predict_ensemble


def run_demo():
    # 1. Setup and Reproducibility
    seed_everything(42)

    # Configure logging to be less verbose for external libraries
    logging.getLogger("timm").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    print("=== Starting Demonstration Script ===")

    # 2. Prepare Demo Environment (Speed Optimization)
    # We create a separate working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Created demo directory: {demo_dir}")

    # 3. Create Subset Metadata (Data Optimization)
    # We filter for a few specific classes to ensure StratifiedKFold and LDA work
    # even with a very small dataset size.
    original_train_path = "./metadata/train.csv"
    original_test_path = "./metadata/test.csv"

    df_train = pd.read_csv(original_train_path)

    # Select top 3 classes by frequency to ensure enough samples per class
    top_classes = df_train["species"].value_counts().head(3).index.tolist()
    df_train_subset = df_train[df_train["species"].isin(top_classes)].copy()

    # Limit to ~30 samples total for speed
    df_train_subset = df_train_subset.head(30)

    # Save subset metadata
    demo_train_path = os.path.join(demo_dir, "train.csv")
    df_train_subset.to_csv(demo_train_path, index=False)
    print(
        f"Created training subset with {len(df_train_subset)} samples (Classes: {top_classes})"
    )

    # Prepare Test Subset
    df_test = pd.read_csv(original_test_path)
    df_test_subset = df_test.head(10).copy()
    demo_test_path = os.path.join(demo_dir, "test.csv")
    df_test_subset.to_csv(demo_test_path, index=False)
    print(f"Created test subset with {len(df_test_subset)} samples")

    # 4. Monkey-patch Config (Runtime Configuration)
    # This redirects the library to use our small datasets and demo directory
    print("Overriding Config for demo execution...")
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.TRAIN_META_PATH = demo_train_path
    Config.TEST_META_PATH = demo_test_path
    Config.N_FOLDS = 2  # Reduce folds for speed
    Config.BATCH_SIZE = 8  # Small batch size

    # 5. Verify Image Processing Logic
    print("\n--- Verifying Image Processing ---")
    processor = ImageProcessor()

    # Pick a valid file from our subset
    sample_file = df_train_subset.iloc[0]["file_path"]
    print(f"Processing sample image: {sample_file}")

    # Process
    views = processor.process_single_file(sample_file)

    # Assertions
    # Expected shape: (4, 224, 224, 3)
    assert isinstance(views, np.ndarray), "Output must be a numpy array"
    assert views.shape == (
        4,
        224,
        224,
        3,
    ), f"Expected shape (4, 224, 224, 3), got {views.shape}"
    assert views.dtype == np.uint8, "Image data should be uint8"
    print("Image Processing verification passed.")

    # 6. Verify Feature Extraction Logic
    print("\n--- Verifying Feature Extraction ---")
    # Initialize extractor (loads models)
    extractor = DualStreamExtractor()

    # Create a dummy batch from the processed view
    # Shape: (1, 4, 224, 224, 3)
    dummy_batch = torch.from_numpy(views).unsqueeze(0)

    # Inference
    dino_emb, conv_emb = extractor.process_batch(dummy_batch)

    # Assertions
    # DINOv2 Large output dim: 1024
    # ConvNeXt Large output dim: 1536
    print(f"DINO Embedding Shape: {dino_emb.shape}")
    print(f"ConvNeXt Embedding Shape: {conv_emb.shape}")

    assert dino_emb.shape == (
        1,
        1024,
    ), f"Expected DINO shape (1, 1024), got {dino_emb.shape}"
    assert conv_emb.shape == (
        1,
        1536,
    ), f"Expected ConvNeXt shape (1, 1536), got {conv_emb.shape}"
    print("Feature Extraction verification passed.")

    # 7. Execute Full Training Pipeline
    print("\n--- Executing Training Pipeline ---")
    # We disable loading cached models to force training on our new subset
    pipelines, label_encoder = train_ensemble(
        load_cached_data=False, load_cached_models=False
    )

    assert (
        len(pipelines) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} pipelines, got {len(pipelines)}"
    print("Training complete.")

    # 8. Execute Inference Pipeline
    print("\n--- Executing Inference Pipeline ---")
    predict_ensemble(pipelines, label_encoder, load_cached_data=False)

    # 9. Verify Submission
    print("\n--- Verifying Submission ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns[:5]}...")

    # Assertions
    # Rows should match test subset size (10)
    # Columns should be id + 99 classes = 100
    assert len(df_sub) == 10, f"Expected 10 rows, got {len(df_sub)}"
    assert df_sub.shape[1] == 100, f"Expected 100 columns, got {df_sub.shape[1]}"

    # Check probability range
    probs = df_sub.drop(columns=["id"]).values
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities must be in [0, 1]"

    print("Submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
