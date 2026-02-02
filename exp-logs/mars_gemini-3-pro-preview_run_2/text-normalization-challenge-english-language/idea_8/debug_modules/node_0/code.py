import os
import pandas as pd
import torch
import shutil
import numpy as np

# Import from provided library
from library.config import cfg
from library.normalization_rules import (
    dispatch_rule,
    IntegerToWords,
    expand_cardinal,
    expand_decimal,
    expand_letters,
)
from library.train_router import run_router_training
from library.train_generator import run_generator_training
from library.inference import run_inference
from library.modeling import RouterModel, GeneratorModel


def setup_demo_environment():
    """
    Sets up a temporary environment with small data subsets to ensure
    the demo runs quickly and verifies logic without processing the full dataset.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_test"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    cfg.WORKING_DIR = demo_dir
    cfg.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    cfg.CACHE_DIR = os.path.join(demo_dir, "cache")
    cfg.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)

    # Override Training Hyperparameters for Speed
    cfg.ROUTER_EPOCHS = 1
    cfg.GENERATOR_EPOCHS = 1
    cfg.ROUTER_BATCH_SIZE = 8
    cfg.GENERATOR_BATCH_SIZE = 8

    # Create Data Subsets (First 2000 rows)
    # This mocks the large dataset with a tiny slice for functional verification
    subset_size = 2000

    print(f"Creating data subsets (Size: {subset_size})...")

    # Train
    df_train = pd.read_csv(
        "./metadata/train.csv", nrows=subset_size, keep_default_na=False
    )
    # Ensure we have some neural classes in the subset for the generator to learn something
    # If the head doesn't have them, we might append some, but for a simple run check, head is usually fine.
    # Let's check if we need to force some diversity.
    if not any(df_train["class"].isin(cfg.NEURAL_BASED_CLASSES)):
        print(
            "Note: Subset lacks neural classes. Generator training might be empty. Proceeding anyway."
        )

    demo_train_path = os.path.join(demo_dir, "train_subset.csv")
    df_train.to_csv(demo_train_path, index=False)
    cfg.TRAIN_FILE = demo_train_path

    # Val
    df_val = pd.read_csv("./metadata/val.csv", nrows=subset_size, keep_default_na=False)
    demo_val_path = os.path.join(demo_dir, "val_subset.csv")
    df_val.to_csv(demo_val_path, index=False)
    cfg.VAL_FILE = demo_val_path

    # Test
    df_test = pd.read_csv(
        "./metadata/test.csv", nrows=subset_size, keep_default_na=False
    )
    demo_test_path = os.path.join(demo_dir, "test_subset.csv")
    df_test.to_csv(demo_test_path, index=False)
    cfg.TEST_FILE = demo_test_path

    print("Configuration updated for demo run.")


def verify_normalization_rules():
    """
    Verifies the deterministic normalization logic.
    """
    print("\n>>> Verifying Normalization Rules...")

    # 1. Integer Conversion
    converter = IntegerToWords()
    assert converter.convert(0) == "zero"
    assert converter.convert(15) == "fifteen"
    assert converter.convert(123) == "one hundred twenty three"
    assert converter.convert(1001) == "one thousand one"
    print("IntegerToWords: OK")

    # 2. Cardinal Expansion
    assert expand_cardinal("500") == "five hundred"
    assert expand_cardinal("-10") == "minus ten"
    print("expand_cardinal: OK")

    # 3. Decimal Expansion
    assert expand_decimal("3.14") == "three point one four"
    assert expand_decimal("0.5") == "zero point five"
    print("expand_decimal: OK")

    # 4. Letters Expansion
    assert expand_letters("FBI") == "f b i"
    assert expand_letters("A&M") == "a and m"
    print("expand_letters: OK")

    # 5. Dispatcher
    assert dispatch_rule("1st", "ORDINAL") == "first"
    assert dispatch_rule("2012", "DIGIT") == "two zero one two"
    assert dispatch_rule("hello", "PLAIN") == "hello"
    print("dispatch_rule: OK")


def verify_router_pipeline():
    """
    Runs the Router (Token Classification) training pipeline.
    """
    print("\n>>> Verifying Router Training Pipeline...")

    # Run training in debug mode (further slices the subset if logic allows,
    # but we already provided a small file, so debug=False in the function call
    # uses our small file as the 'full' dataset).
    # We set debug=True to trigger the internal slicer just to be safe and fast.
    run_router_training(debug=True, load_cached_data=False)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "router_best")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Router checkpoint not found at {checkpoint_path}")

    print("Router training completed and checkpoint saved.")


def verify_generator_pipeline():
    """
    Runs the Generator (Seq2Seq) training pipeline.
    """
    print("\n>>> Verifying Generator Training Pipeline...")

    # Run training
    run_generator_training(debug=True, load_cached_data=False)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "generator_best")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found at {checkpoint_path}")

    print("Generator training completed and checkpoint saved.")


def verify_inference_pipeline():
    """
    Runs the full inference pipeline using the trained models.
    """
    print("\n>>> Verifying Inference Pipeline...")

    # Run inference
    run_inference()

    # Verify submission file
    submission_path = os.path.join(cfg.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    # Check format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "after" in df_sub.columns, "Submission missing 'after' column"

    # Check if rows match test input size (subset size)
    # Note: df_test was subset to 2000 rows.
    # The inference pipeline uses cfg.TEST_FILE which we pointed to the subset.
    df_test_input = pd.read_csv(cfg.TEST_FILE)
    assert len(df_sub) == len(
        df_test_input
    ), f"Submission row count mismatch: {len(df_sub)} vs {len(df_test_input)}"

    # Spot check a prediction
    print("Sample predictions:")
    print(df_sub.head())

    print("Inference pipeline verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    cfg.seed_everything()

    # 1. Setup
    setup_demo_environment()

    # 2. Verify Logic
    verify_normalization_rules()

    # 3. Verify Training
    # We use try-except blocks here only to ensure one failure doesn't stop the whole demo
    # if we wanted to be lenient, but the prompt says "All validation checks must fail explicitly".
    # So we will run them sequentially.

    verify_router_pipeline()
    verify_generator_pipeline()

    # 4. Verify Inference
    verify_inference_pipeline()

    print("\n>>> All demonstrations completed successfully.")
