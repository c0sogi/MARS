import os
import shutil
import torch
import pandas as pd
import numpy as np
import logging
from transformers import logging as transformers_logging

# Import library modules
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.feature_extraction import FeatureExtractor
from library.data_loader import get_dataloader
from library.model import DCCodeBERT
from library.trainer import Trainer


def run_demo():
    print("=== Starting AI4Code Solution Demo ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Override
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Set seeds for reproducibility
    set_seed(42)

    # Suppress verbose warnings from libraries
    transformers_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config for a fast, small-scale run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduced workers

    # Define a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to the demo directory
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    Config.print_config()

    # ---------------------------------------------------------
    # 2. Feature Extraction
    # ---------------------------------------------------------
    print("\n[2] Running Feature Extraction...")
    # This will load raw JSONs, tokenize, run CodeBERT, and save embeddings to Parquet
    extractor = FeatureExtractor()
    extractor.extract_and_save_features(load_cached_data=False)

    # Verify features were created
    assert os.path.exists(
        Config.TRAIN_FEATURES_PATH
    ), "Train features parquet not found!"
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features parquet not found!"
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test features parquet not found!"

    # Check content of one file
    df_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
    print(f"Train features shape: {df_train.shape}")
    assert "embedding" in df_train.columns, "Embeddings missing from train features"
    assert not df_train.empty, "Train features dataframe is empty"

    # ---------------------------------------------------------
    # 3. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loader...")
    # Create a dataloader to inspect a batch
    train_loader = get_dataloader(
        Config.TRAIN_FEATURES_PATH, batch_size=2, shuffle=True, mode="train"
    )

    batch = next(iter(train_loader))

    # Inspect batch keys
    required_keys = [
        "id",
        "code_embeddings",
        "code_mask",
        "md_embeddings",
        "md_mask",
        "labels",
        "md_ids",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check shapes
    # code_embeddings: (Batch, Seq_Len, Hidden)
    B = len(batch["id"])
    assert batch["code_embeddings"].dim() == 3
    assert batch["code_embeddings"].size(0) == B
    assert batch["code_embeddings"].size(2) == Config.HIDDEN_DIM

    # md_embeddings: (Batch, Seq_Len, Hidden)
    assert batch["md_embeddings"].dim() == 3
    assert batch["md_embeddings"].size(0) == B

    # labels: (Batch, Seq_Len)
    assert batch["labels"].dim() == 2
    assert batch["labels"].size(0) == B

    print("Data Loader verification passed. Batch shapes look correct.")

    # ---------------------------------------------------------
    # 4. Model Training and Inference
    # ---------------------------------------------------------
    print("\n[4] Running Trainer (Train -> Val -> Test Prediction)...")
    # Trainer handles the full loop: loading data, training model, validating, and generating submission
    trainer = Trainer()
    trainer.fit()

    # ---------------------------------------------------------
    # 5. Output Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Outputs...")

    # Check if model checkpoint exists
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint not found!"

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    assert list(df_sub.columns) == [
        "id",
        "cell_order",
    ], "Submission columns are incorrect"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check that cell_order is a string of space-separated IDs
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order is not a string"
    assert len(sample_order.split()) > 0, "cell_order seems empty"

    print("Pipeline execution successful.")

    # ---------------------------------------------------------
    # 6. Metric Logic Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Metric Logic (Kendall Tau)...")

    # Case 1: Perfect match
    # Ground Truth: A B C D
    # Prediction:   A B C D
    gt_1 = ["A", "B", "C", "D"]
    pred_1 = ["A", "B", "C", "D"]

    # Case 2: Complete reversal
    # Ground Truth: A B C D
    # Prediction:   D C B A
    # n=4, pairs = 4*3 = 12.
    # Inversions: (D,C), (D,B), (D,A), (C,B), (C,A), (B,A) = 6 inversions.
    # Score = 1 - 4 * (6 / 12) = 1 - 2 = -1.0
    gt_2 = ["A", "B", "C", "D"]
    pred_2 = ["D", "C", "B", "A"]

    score_perfect = compute_kendall_tau([pred_1], [gt_1])
    score_reversed = compute_kendall_tau([pred_2], [gt_2])

    print(f"Score (Perfect Match): {score_perfect}")
    print(f"Score (Reversed): {score_reversed}")

    assert abs(score_perfect - 1.0) < 1e-6, "Metric logic failed for perfect match"
    assert abs(score_reversed - (-1.0)) < 1e-6, "Metric logic failed for reversed match"

    print("Metric verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
