import os
import sys
import shutil
import pandas as pd
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ------------------------------------------------------------------------------
# 1. Environment Setup & Configuration Override
# ------------------------------------------------------------------------------
# Disable tqdm progress bars as per requirements
os.environ["TQDM_DISABLE"] = "1"

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_data, LocatorDataset, InfillerDataset, TestDataset
from library.models import ModelFactory
from library.engine import Trainer
from library.inference import BeamPipeline


def setup_demo_config():
    """
    Overrides the default Config class attributes for a fast demonstration run.
    """
    print(">>> Setting up demo configuration...")

    # Define a separate working directory for the demo to avoid conflicts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override General Config
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Data Sampling (Tiny subset for speed)
    Config.TRAIN_SAMPLE_SIZE = 50
    Config.VAL_SAMPLE_SIZE = 10

    # Override Training Hyperparameters (1 epoch, small batch)
    Config.LOCATOR_EPOCHS = 1
    Config.INFILLER_EPOCHS = 1
    Config.LOCATOR_BATCH_SIZE = 4
    Config.INFILLER_BATCH_SIZE = 4

    # Override Paths for Artifacts
    Config.LOCATOR_TRAIN_CACHE = os.path.join(demo_dir, "locator_train.parquet")
    Config.LOCATOR_VAL_CACHE = os.path.join(demo_dir, "locator_val.parquet")
    Config.INFILLER_TRAIN_CACHE = os.path.join(demo_dir, "infiller_train.parquet")
    Config.INFILLER_VAL_CACHE = os.path.join(demo_dir, "infiller_val.parquet")

    Config.LOCATOR_MODEL_PATH = os.path.join(demo_dir, "best_locator.pth")
    Config.INFILLER_MODEL_PATH = os.path.join(demo_dir, "best_infiller.pth")

    # Create necessary directories
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(">>> Configuration updated successfully.")


# ------------------------------------------------------------------------------
# 2. Data Processing Demonstration
# ------------------------------------------------------------------------------
def demo_data_processing():
    print("\n>>> Starting Data Processing Demo...")

    # Run the processing pipeline
    # load_cached_data=False ensures we actually run the logic
    loc_train, loc_val, inf_train, inf_val = process_data(load_cached_data=False)

    # Validation
    print(f"Locator Train Size: {len(loc_train)}")
    print(f"Infiller Train Size: {len(inf_train)}")

    assert len(loc_train) > 0, "Locator training set is empty."
    assert len(inf_train) > 0, "Infiller training set is empty."
    assert "gap_index" in loc_train.columns, "Locator data missing 'gap_index'."
    assert "missing_word" in inf_train.columns, "Infiller data missing 'missing_word'."

    print(">>> Data Processing verified.")
    return loc_train, loc_val, inf_train, inf_val


# ------------------------------------------------------------------------------
# 3. Model Training Demonstration
# ------------------------------------------------------------------------------
def demo_training(loc_train, loc_val, inf_train, inf_val):
    print("\n>>> Starting Training Demo...")

    # Initialize Tokenizers
    print("Loading tokenizers...")
    loc_tokenizer = AutoTokenizer.from_pretrained(
        Config.LOCATOR_MODEL_NAME, use_fast=True
    )
    inf_tokenizer = AutoTokenizer.from_pretrained(
        Config.INFILLER_MODEL_NAME, use_fast=True
    )

    # Create Datasets
    loc_train_ds = LocatorDataset(loc_train, loc_tokenizer)
    loc_val_ds = LocatorDataset(loc_val, loc_tokenizer)

    inf_train_ds = InfillerDataset(inf_train, inf_tokenizer)
    inf_val_ds = InfillerDataset(inf_val, inf_tokenizer)

    # Create DataLoaders
    loc_train_loader = DataLoader(
        loc_train_ds, batch_size=Config.LOCATOR_BATCH_SIZE, shuffle=True
    )
    loc_val_loader = DataLoader(loc_val_ds, batch_size=Config.LOCATOR_BATCH_SIZE)

    inf_train_loader = DataLoader(
        inf_train_ds, batch_size=Config.INFILLER_BATCH_SIZE, shuffle=True
    )
    inf_val_loader = DataLoader(inf_val_ds, batch_size=Config.INFILLER_BATCH_SIZE)

    # Initialize Trainer
    trainer = Trainer()

    # Train Locator
    print("Training Locator...")
    trainer.train_locator(loc_train_loader, loc_val_loader)
    assert os.path.exists(
        Config.LOCATOR_MODEL_PATH
    ), "Locator model checkpoint not found."

    # Train Infiller
    print("Training Infiller...")
    trainer.train_infiller(inf_train_loader, inf_val_loader)
    assert os.path.exists(
        Config.INFILLER_MODEL_PATH
    ), "Infiller model checkpoint not found."

    print(">>> Training verified.")


# ------------------------------------------------------------------------------
# 4. Inference Demonstration
# ------------------------------------------------------------------------------
def demo_inference():
    print("\n>>> Starting Inference Demo...")

    # Create a small synthetic test set for inference
    # We create a new parquet file to simulate the test set structure
    # Sentence: "The quick brown fox over the lazy dog." -> Missing "jumps"
    # Input: "The quick brown fox over the lazy dog ."

    test_data = [
        {"id": 1, "sentence": "The quick brown fox over the lazy dog ."},
        {"id": 2, "sentence": "She sells sea by the sea shore ."},
    ]
    df_test = pd.DataFrame(test_data)

    # Save to a temporary location and point Config to it
    temp_test_path = os.path.join(Config.WORKING_DIR, "demo_test.parquet")
    df_test.to_parquet(temp_test_path)
    Config.TEST_METADATA_PATH = temp_test_path

    # Initialize Pipeline
    pipeline = BeamPipeline()

    # Create Test DataLoader
    # Note: BeamPipeline handles its own tokenization internally for offsets,
    # but TestDataset requires a tokenizer for basic collation.
    dummy_tokenizer = AutoTokenizer.from_pretrained(Config.LOCATOR_MODEL_NAME)
    test_dataset = TestDataset(df_test, dummy_tokenizer)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.LOCATOR_BATCH_SIZE, shuffle=False
    )

    # Run Prediction
    results = pipeline.predict(test_loader)

    # Verify Results
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"Sample Prediction: ID={results[0][0]}, Sentence='{results[0][1]}'")

    # Generate Submission
    pipeline.generate_submission(results)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    assert len(lines) == 3, f"Expected header + 2 rows, got {len(lines)} lines."
    assert lines[0].strip() == 'id,"sentence"', "Header mismatch."

    print(">>> Inference verified.")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure Logging to be less verbose for the demo
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("data").setLevel(logging.INFO)
    logging.getLogger("engine").setLevel(logging.INFO)
    logging.getLogger("inference").setLevel(logging.INFO)

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Data
        loc_train, loc_val, inf_train, inf_val = demo_data_processing()

        # 3. Training
        demo_training(loc_train, loc_val, inf_train, inf_val)

        # 4. Inference
        demo_inference()

        print("\n>>> DEMO COMPLETED SUCCESSFULLY.")

    except AssertionError as e:
        print(f"\n!!! DEMO FAILED: Assertion Error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! DEMO FAILED: Exception - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
