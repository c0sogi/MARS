import pandas as pd
import torch
import os
import sys
import shutil
import numpy as np

# Import library components
from library.config import Config
from library.normalization_rules import Normalizer
from library.dataset import TextNormalizationDataset
from library.model import TokenClassifier
from library.trainer import Trainer
from library.inference import InferencePipeline


def run_demo():
    print("=== Starting Text Normalization Library Demo ===")

    # ==========================================
    # 1. Setup Environment and Config Overrides
    # ==========================================
    # Create a temporary working directory for this demo run
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model_checkpoint.bin")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    # Ensure cache dir exists inside new working dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Create a consistent small subset of test data for alignment verification
    # We select the first 50 sentences (by ID) to match the debug_size used later
    print("Creating temporary aligned test set...")
    full_test_df = pd.read_csv("./metadata/test.csv", keep_default_na=False)

    # Sort by sentence_id to ensure deterministic selection matching dataset.py logic
    unique_sents = sorted(full_test_df["sentence_id"].unique())[:50]
    subset_test_df = full_test_df[full_test_df["sentence_id"].isin(unique_sents)].copy()

    # Save this subset to a temporary file and point Config to it
    temp_test_path = os.path.join(demo_dir, "temp_test.csv")
    subset_test_df.to_csv(temp_test_path, index=False)
    Config.TEST_DATA_PATH = temp_test_path

    print(f"Test subset saved to {temp_test_path} ({len(subset_test_df)} tokens)")

    # ==========================================
    # 2. Demonstrate Normalizer
    # ==========================================
    print("\n--- Testing Normalizer ---")
    norm = Normalizer()

    # Verify Money Normalization
    raw_money = "$3.16"
    norm_money = norm.normalize(raw_money, "MONEY")
    print(f"Rule Check: '{raw_money}' (MONEY) -> '{norm_money}'")
    assert (
        norm_money == "three dollars, sixteen cents"
    ), f"Expected 'three dollars, sixteen cents', got '{norm_money}'"

    # Verify Date Normalization
    raw_date = "2012"
    norm_date = norm.normalize(raw_date, "DATE")
    print(f"Rule Check: '{raw_date}' (DATE) -> '{norm_date}'")
    assert norm_date == "twenty twelve", f"Expected 'twenty twelve', got '{norm_date}'"

    # Verify Plain (Identity)
    assert norm.normalize("test", "PLAIN") == "test"

    print("Normalizer logic verified.")

    # ==========================================
    # 3. Demonstrate Dataset Loading
    # ==========================================
    print("\n--- Testing Dataset ---")
    # Initialize dataset with debug_size=50 (sentences)
    # We disable caching to ensure it reads our fresh config/files
    train_ds = TextNormalizationDataset(
        split="train", debug_size=50, load_cached_data=False
    )

    print(f"Train Dataset Size: {len(train_ds)} sentences")
    assert (
        len(train_ds) == 50
    ), "Dataset should contain exactly 50 sentences based on debug_size"

    # Inspect a single item
    item = train_ds[0]
    print("Keys in dataset item:", list(item.keys()))

    # Verify tensor structures
    assert "input_ids" in item
    assert "labels" in item
    assert "attention_mask" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].shape[0] == Config.MAX_LEN

    print("Dataset structure verified.")

    # ==========================================
    # 4. Demonstrate Model
    # ==========================================
    print("\n--- Testing Model ---")
    model = TokenClassifier()
    model.to(Config.DEVICE)

    # Prepare a batch (unsqueeze to add batch dimension)
    input_ids = item["input_ids"].unsqueeze(0).to(Config.DEVICE)
    attention_mask = item["attention_mask"].unsqueeze(0).to(Config.DEVICE)

    # Run forward pass
    output = model(input_ids, attention_mask)
    logits = output.logits

    print(f"Logits Shape: {logits.shape}")
    expected_shape = (1, Config.MAX_LEN, Config.NUM_LABELS)
    assert (
        logits.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {logits.shape}"

    print("Model forward pass verified.")

    # ==========================================
    # 5. Demonstrate Trainer (Training & Submission)
    # ==========================================
    print("\n--- Testing Trainer ---")
    trainer = Trainer()

    # Train for 1 epoch using the small dataset for both train and val
    print("Running training loop (1 epoch)...")
    trainer.train(train_ds, train_ds, epochs=1)

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("Model checkpoint saved successfully.")

    # Generate Submission
    # We use a test dataset initialized with debug_size=50.
    # Because we pointed Config.TEST_DATA_PATH to a file with exactly 50 sentences,
    # the predictions and the raw text file will align perfectly.
    print("Generating submission...")
    test_ds = TextNormalizationDataset(
        split="test", debug_size=50, load_cached_data=False
    )
    trainer.generate_submission(test_ds)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission rows: {len(sub_df)}")

    # The submission rows should match the number of tokens in our subset_test_df
    assert len(sub_df) == len(
        subset_test_df
    ), f"Submission length {len(sub_df)} mismatch with test data {len(subset_test_df)}"

    print("Submission generation verified.")

    # ==========================================
    # 6. Demonstrate InferencePipeline
    # ==========================================
    print("\n--- Testing InferencePipeline ---")
    # Initialize pipeline with the model we just trained
    pipeline = InferencePipeline(model_path=Config.MODEL_SAVE_PATH)

    # Predict classes for the test dataset
    preds = pipeline.predict_classes(test_ds)

    print(f"Inference generated predictions for {len(preds)} sentences.")
    assert len(preds) == 50, "Inference should return predictions for 50 sentences"

    print("InferencePipeline verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
