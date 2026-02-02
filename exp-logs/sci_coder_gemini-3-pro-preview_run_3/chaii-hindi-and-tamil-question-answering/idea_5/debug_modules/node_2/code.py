import os
import shutil
import pandas as pd
import torch
import transformers
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.tapt_engine import run_tapt
from library.data_factory import get_dataloader
from library.qa_engine import run_training
from library.inference_engine import InferenceEngine

# Suppress warnings and verbose logs for cleaner output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_demo_config():
    """
    Monkey-patches the Config class to use a temporary directory and
    minimal hyperparameters for a fast demonstration.
    """
    print("Setting up demonstration configuration...")

    # 1. Redirect paths to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.QA_CACHE_DIR = os.path.join(Config.WORKING_DIR, "qa_cache")
    Config.TAPT_CACHE_DIR = os.path.join(Config.WORKING_DIR, "tapt_cache")
    Config.QA_MODEL_DIR = os.path.join(Config.WORKING_DIR, "qa_models")
    Config.TAPT_MODEL_DIR = os.path.join(Config.WORKING_DIR, "tapt_model_finetuned")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # 2. Reduce hyperparameters for speed
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.TAPT_EPOCHS = 1  # TAPT for only 1 epoch
    Config.SEEDS = [42]  # Run only one seed
    Config.TRAIN_BATCH_SIZE = 4  # Small batch size
    Config.EVAL_BATCH_SIZE = 8
    Config.TAPT_BATCH_SIZE = 4

    # 3. Create the new directory structure
    # Re-run setup to create these folders
    Config.setup()

    print(f"Demo Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Seeds: {Config.SEEDS}")


def demo_tapt_pipeline():
    """
    Demonstrates the Task-Adaptive Pretraining (TAPT) pipeline.
    """
    print("\n=== Running TAPT Pipeline Demo ===")

    # Run TAPT
    # We disable loading from cache to force the logic to run
    run_tapt(
        load_cached_data=False,
        batch_size=Config.TAPT_BATCH_SIZE,
        num_epochs=Config.TAPT_EPOCHS,
    )

    # Verify outputs
    expected_files = ["config.json", "model.safetensors", "tokenizer.json"]
    for fname in expected_files:
        fpath = os.path.join(Config.TAPT_MODEL_DIR, fname)
        assert os.path.exists(fpath), f"TAPT output file missing: {fpath}"

    print("TAPT Pipeline verification successful.")


def demo_data_loading():
    """
    Demonstrates and verifies the Data Factory and DataLoader logic.
    """
    print("\n=== Running Data Loading Demo ===")

    # Get a dataloader for the training set
    # Using a small batch size to inspect tensor shapes
    batch_size = 2
    dataloader = get_dataloader(
        mode="train", batch_size=batch_size, load_cached_data=False
    )

    # Fetch one batch
    batch = next(iter(dataloader))

    # Verify keys
    expected_keys = [
        "input_ids",
        "attention_mask",
        "labels",
        "offset_mapping",
        "example_id",
    ]
    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify shapes
    # input_ids: (batch_size, max_seq_len)
    assert batch["input_ids"].shape == (
        batch_size,
        Config.MAX_LENGTH,
    ), f"Incorrect input_ids shape: {batch['input_ids'].shape}"

    assert batch["labels"].shape == (
        batch_size,
        Config.MAX_LENGTH,
    ), f"Incorrect labels shape: {batch['labels'].shape}"

    print(f"Batch loaded successfully. Input shape: {batch['input_ids'].shape}")
    print("Data Loading verification successful.")


def demo_qa_training():
    """
    Demonstrates the QA Fine-tuning pipeline.
    """
    print("\n=== Running QA Training Demo ===")

    # Run training (uses the TAPT model generated in the previous step if available)
    run_training(load_cached_data=True)

    # Verify model checkpoint creation
    seed = Config.SEEDS[0]
    model_path = os.path.join(Config.QA_MODEL_DIR, f"model_seed_{seed}.pt")

    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"

    # Verify model size (basic check to ensure it's not empty)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model saved at {model_path} ({file_size_mb:.2f} MB)")
    assert file_size_mb > 100, "Model file seems too small to be valid."

    print("QA Training verification successful.")


def demo_inference():
    """
    Demonstrates the Inference pipeline and submission generation.
    """
    print("\n=== Running Inference Demo ===")

    engine = InferenceEngine()

    # Generate submission
    engine.generate_submission(load_cached_data=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Cite debug_lesson_7: Disable default NA parsing to handle valid empty strings correctly
    df_sub = pd.read_csv(Config.SUBMISSION_PATH, keep_default_na=False)

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing 'PredictionString' column"

    # Check row count (Test set has 112 rows)
    assert len(df_sub) == 112, f"Expected 112 predictions, got {len(df_sub)}"

    # Check content (PredictionString should be string)
    assert (
        df_sub["PredictionString"].dtype == object
    ), f"PredictionString column should be object/string, got {df_sub['PredictionString'].dtype}"

    print("Head of generated submission:")
    print(df_sub.head())
    print("Inference verification successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        # 1. Setup Config
        setup_demo_config()

        # 2. Run TAPT Demo
        demo_tapt_pipeline()

        # 3. Run Data Loading Demo
        demo_data_loading()

        # 4. Run QA Training Demo
        demo_qa_training()

        # 5. Run Inference Demo
        demo_inference()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
