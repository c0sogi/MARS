import os
import sys
import pandas as pd
import torch
import shutil
import warnings
import logging

# Suppress warnings and logging for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import library modules
from library.config import Config
from library.utils import set_seed, jaccard
from library.tapt_engine import run_tapt
from library.trainer import run_training
from library.inference import run_inference
from library.data_loader import prepare_features


class DemoConfig(Config):
    """
    Custom configuration for the demonstration run.
    Overrides paths to use a separate working directory and reduces
    computational load (epochs, batch sizes) for speed.
    """

    def __init__(self):
        super().__init__()
        # Override working directory
        self.working_dir = "./working/demo_run"

        # Update sub-directories based on new working_dir
        self.metadata_dir = os.path.join(self.working_dir, "metadata")
        self.cache_dir = os.path.join(self.working_dir, "qa_cache")
        self.model_dir = os.path.join(self.working_dir, "qa_models")
        self.submission_dir = os.path.join(self.working_dir, "submission")
        self.tapt_cache_dir = os.path.join(self.working_dir, "tapt_cache")
        self.tapt_output_dir = os.path.join(self.working_dir, "tapt_model_finetuned")

        # Create directories
        for d in [
            self.working_dir,
            self.metadata_dir,
            self.cache_dir,
            self.model_dir,
            self.submission_dir,
            self.tapt_cache_dir,
            self.tapt_output_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # Reduce hyperparameters for speed
        self.epochs = 1
        self.batch_size = 4
        self.seeds = [42]  # Run only one seed
        self.num_workers = 0  # Avoid multiprocessing overhead for small data

    def get_tapt_config(self):
        """Override TAPT config for speed."""
        conf = super().get_tapt_config()
        conf["num_train_epochs"] = 1
        conf["train_batch_size"] = 4
        return conf


def create_subset_metadata(config: DemoConfig, num_samples=20):
    """
    Creates a small subset of the original metadata files to ensure
    the demo runs quickly.
    """
    print(f"Creating data subsets (n={num_samples}) in {config.metadata_dir}...")

    # Source directory for original metadata
    original_meta_dir = "./metadata"

    for split in ["train.csv", "val.csv", "test.csv"]:
        src_path = os.path.join(original_meta_dir, split)
        dst_path = os.path.join(config.metadata_dir, split)

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Take a subset
            df_subset = df.head(num_samples)
            df_subset.to_csv(dst_path, index=False)
            print(f"  Created {split} subset with {len(df_subset)} rows.")
        else:
            raise FileNotFoundError(f"Original metadata {src_path} not found.")


def verify_jaccard():
    """Verifies the jaccard utility function."""
    s1 = "machine learning is fun"
    s2 = "learning is fun"
    # Intersection: "learning", "is", "fun" (3)
    # Union: "machine", "learning", "is", "fun" (4)
    # Score: 3/4 = 0.75
    score = jaccard(s1, s2)
    assert (
        abs(score - 0.75) < 1e-6
    ), f"Jaccard calculation failed. Expected 0.75, got {score}"
    print("Jaccard metric verification passed.")


def main():
    # 1. Setup Configuration
    print("--- 1. Initializing Configuration ---")
    config = DemoConfig()
    set_seed(config.seed)

    # 2. Prepare Data Subset
    # We use a subset of the data to make this script run within the time limit
    create_subset_metadata(config, num_samples=20)

    # 3. Verify Utility
    print("\n--- 2. Verifying Utilities ---")
    verify_jaccard()

    # 4. Run TAPT (Task-Adaptive Pretraining)
    # This demonstrates using the TAPT engine to fine-tune the language model on the corpus
    print("\n--- 3. Running TAPT Pipeline ---")
    run_tapt(config, load_cached_data=False)

    # Verify TAPT output
    assert os.path.exists(
        os.path.join(config.tapt_output_dir, "model.safetensors")
    ) or os.path.exists(
        os.path.join(config.tapt_output_dir, "pytorch_model.bin")
    ), "TAPT model file not generated."

    # 5. Run QA Training
    # This demonstrates feature preparation, model loading (from TAPT), and training loop
    print("\n--- 4. Running QA Training Pipeline ---")
    run_training(config)

    # Verify QA Model output
    model_path = os.path.join(config.model_dir, f"model_seed_{config.seeds[0]}.pt")
    assert os.path.exists(model_path), f"QA Model checkpoint not found at {model_path}"

    # 6. Run Inference
    # This demonstrates loading the trained model, predicting on test set, and ensembling
    print("\n--- 5. Running Inference Pipeline ---")
    run_inference(config)

    # 7. Verify Submission
    submission_path = os.path.join(config.submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found."

    df_sub = pd.read_csv(submission_path)
    print(f"\nSubmission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())

    # Check format
    expected_cols = ["id", "PredictionString"]
    assert list(df_sub.columns) == expected_cols, f"Invalid columns: {df_sub.columns}"
    assert len(df_sub) > 0, "Submission file is empty."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
