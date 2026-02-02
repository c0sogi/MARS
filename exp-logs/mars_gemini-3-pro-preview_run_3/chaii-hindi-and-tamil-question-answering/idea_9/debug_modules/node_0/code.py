import os
import pandas as pd
import torch
import shutil
import warnings

# Import library components
from library.configuration import Config
from library.utils import seed_everything, jaccard, clean_text
from library.tapt_manager import run_tapt_training
from library.qa_trainer import train_model
from library.inference_manager import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data(source_dir, target_dir, n_rows=20):
    """
    Reads original metadata, samples n_rows, and saves to target_dir.
    This ensures the demo runs quickly on a small dataset.
    """
    print(f"Creating data subsets (n={n_rows}) in {target_dir}...")

    splits = ["train.csv", "val.csv", "test.csv"]
    for filename in splits:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Sample subset, handling cases where df might be smaller than n_rows
            subset_df = df.head(min(len(df), n_rows))
            subset_df.to_csv(dst_path, index=False)
            print(f"  Saved {filename}: {len(subset_df)} rows")
        else:
            print(f"  Warning: {filename} not found in source.")


def verify_utilities():
    """
    Verifies the correctness of utility functions.
    """
    print("Verifying utility functions...")

    # Test Jaccard
    s1 = "This is a test"
    s2 = "This is a test"
    assert jaccard(s1, s2) == 1.0, "Jaccard should be 1.0 for identical strings"

    s1 = "This is a test"
    s2 = "This is a different test"
    # Intersection: {this, is, a, test} (4)
    # Union: {this, is, a, test, different} (5)
    # Score: 4/5 = 0.8
    score = jaccard(s1, s2)
    assert (
        abs(score - 0.8) < 1e-6
    ), f"Jaccard calculation error. Expected 0.8, got {score}"

    s1 = "completely different"
    s2 = "totally unique"
    assert jaccard(s1, s2) == 0.0, "Jaccard should be 0.0 for disjoint strings"

    # Test Clean Text
    raw = "  Some text with spaces  "
    assert (
        clean_text(raw) == "Some text with spaces"
    ), "clean_text failed to strip whitespace"
    assert clean_text(None) == "", "clean_text failed to handle None"

    print("Utility verification passed.")


def configure_demo_environment():
    """
    Overrides Config parameters for a fast demonstration run.
    """
    print("Configuring demo environment...")

    # Define new working directory for the demo
    demo_dir = "./working/demo_run"
    meta_dir = os.path.join(demo_dir, "metadata")

    # Ensure directories exist
    os.makedirs(meta_dir, exist_ok=True)

    # Create subsets
    create_subset_data("./metadata", meta_dir, n_rows=20)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_META_PATH = os.path.join(meta_dir, "train.csv")
    Config.VAL_META_PATH = os.path.join(meta_dir, "val.csv")
    Config.TEST_META_PATH = os.path.join(meta_dir, "test.csv")

    # Update output directories based on the new working dir
    Config.QA_CACHE_DIR = os.path.join(Config.WORKING_DIR, "qa_cache")
    Config.QA_MODELS_DIR = os.path.join(Config.WORKING_DIR, "qa_models")
    Config.TAPT_CACHE_DIR = os.path.join(Config.WORKING_DIR, "tapt_cache")
    Config.TAPT_OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "tapt_model_finetuned")
    Config.TAPT_CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "tapt_checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create these directories
    Config.setup()

    # Override Hyperparameters for speed
    Config.EPOCHS = 1
    Config.TAPT_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TAPT_BATCH_SIZE = 4
    Config.SEEDS = [42]  # Only run one seed
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Set device to GPU if available
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {Config.DEVICE}")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    configure_demo_environment()
    verify_utilities()

    # 2. Run TAPT (Task-Adaptive Pretraining)
    # This fine-tunes the language model on the unlabelled text corpus
    print("\n=== Step 1: Running TAPT ===")
    # load_cached_data=False ensures we process our new subset data, not old cache
    tapt_model_path = run_tapt_training(load_cached_data=False)

    # Verify TAPT output
    assert os.path.exists(tapt_model_path), "TAPT output directory not found"
    assert os.path.exists(
        os.path.join(tapt_model_path, "pytorch_model.bin")
    ) or os.path.exists(
        os.path.join(tapt_model_path, "model.safetensors")
    ), "TAPT model weights not found"
    print(f"TAPT completed successfully. Model saved to {tapt_model_path}")

    # 3. Run QA Training
    # This fine-tunes the TAPT model on the Question-Answering task
    print("\n=== Step 2: Running QA Training ===")
    # We use the output of TAPT as the starting point
    # In a real run, we might loop over Config.SEEDS, but here we just use the first one
    seed = Config.SEEDS[0]
    qa_model_path = train_model(
        seed=seed, pretrained_path=tapt_model_path, load_cached_data=False
    )

    # Verify QA output
    assert os.path.exists(qa_model_path), f"QA model file not found at {qa_model_path}"
    print(f"QA Training completed successfully. Best model saved to {qa_model_path}")

    # 4. Run Inference
    # Generate predictions on the test set
    print("\n=== Step 3: Running Inference ===")
    submission_path = generate_submission(
        model_paths=[qa_model_path], load_cached_data=False
    )

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["id", "PredictionString"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Invalid columns. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check that we have rows (based on our subset size)
    # Note: We subsetted test.csv to 20 rows (or less if original was smaller)
    assert len(df_sub) > 0, "Submission file is empty"

    # Check value types
    assert df_sub["id"].dtype == object, "ID column should be object/string"

    print("\n=== Demo Completed Successfully ===")
    print(f"Final submission located at: {submission_path}")
