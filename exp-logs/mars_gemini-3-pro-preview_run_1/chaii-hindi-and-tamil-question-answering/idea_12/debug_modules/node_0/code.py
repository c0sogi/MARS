import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import transformers
from torch.utils.data import Subset

# Import library modules
from library.config import Config
from library.utils import set_seed, jaccard
from library.data import get_data
from library.model import XLMRobertaForQA
from library.trainer import train_model
from library.inference import generate_submission


# =============================================================================
# SETUP & CONFIGURATION
# =============================================================================
def setup_environment():
    """Sets up the environment for the run."""
    # Suppress verbose warnings for cleaner output
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set seed for reproducibility
    set_seed(42)


def main():
    print("==== Starting Library Demonstration ====")
    setup_environment()

    # 1. Initialize Configuration
    # We use debug=True to set reduced defaults, then further optimize for speed.
    print("\n[1] Initializing Configuration...")
    cfg = Config(debug=True)

    # Override for extreme speed in this demo
    cfg.epochs = 1
    cfg.train_batch_size = 2
    cfg.valid_batch_size = 4
    # Ensure we use the provided metadata directory
    cfg.metadata_dir = "./metadata"
    # Use a specific working directory for this demo
    cfg.working_dir = "./working/demo_run"
    cfg.cache_dir = os.path.join(cfg.working_dir, "cache")
    cfg.output_dir = os.path.join(cfg.working_dir, "output")
    cfg.submission_dir = os.path.join(cfg.working_dir, "submission")

    # Create necessary directories
    os.makedirs(cfg.working_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.submission_dir, exist_ok=True)

    print(f"    Working Directory: {cfg.working_dir}")
    print(f"    Device: {cfg.device}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\n[2] Loading and Processing Data...")
    # We force load_cached_data=False to demonstrate the processing logic at least once.
    # Subsequent calls (like in inference) will use the cache generated here.
    train_dataset, test_dataset, test_features = get_data(cfg, load_cached_data=False)

    # Verification
    print(f"    Train Dataset Size: {len(train_dataset)}")
    print(f"    Test Dataset Size: {len(test_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty!")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty!")

    # Inspect a single sample
    sample = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "relevance_labels",
    ]
    for key in required_keys:
        if key not in sample:
            raise AssertionError(f"Missing key '{key}' in dataset sample.")
    print("    Data integrity check passed.")

    # =========================================================================
    # MODEL INITIALIZATION & VERIFICATION
    # =========================================================================
    print("\n[3] Initializing Model...")
    model = XLMRobertaForQA(cfg.model_name)
    model.to(cfg.device)
    model.eval()

    # Create a dummy batch to verify forward pass
    print("    Verifying forward pass...")
    dummy_input_ids = sample["input_ids"].unsqueeze(0).to(cfg.device)
    dummy_mask = sample["attention_mask"].unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        outputs = model(dummy_input_ids, attention_mask=dummy_mask)

    # Check outputs
    if (
        "start_logits" not in outputs
        or "end_logits" not in outputs
        or "relevance_logits" not in outputs
    ):
        raise AssertionError("Model output missing required logits keys.")

    # Check shapes
    seq_len = dummy_input_ids.shape[1]
    assert outputs["start_logits"].shape == (
        1,
        seq_len,
    ), f"Shape mismatch: {outputs['start_logits'].shape}"
    assert outputs["relevance_logits"].shape == (
        1,
    ), f"Shape mismatch: {outputs['relevance_logits'].shape}"

    print("    Model forward pass successful.")
    del model, dummy_input_ids, dummy_mask, outputs
    torch.cuda.empty_cache()

    # =========================================================================
    # TRAINING DEMONSTRATION
    # =========================================================================
    print("\n[4] Running Training Loop (Subset)...")

    # Create a tiny subset of the training data to ensure this finishes in seconds
    # We use 10 samples.
    subset_indices = list(range(min(10, len(train_dataset))))
    train_subset = Subset(train_dataset, subset_indices)

    print(f"    Training on {len(train_subset)} samples for 1 epoch...")
    train_model(cfg, train_subset)

    # Verify model was saved
    expected_model_path = os.path.join(cfg.output_dir, "model_seed_42.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(f"Model file was not created at {expected_model_path}")
    print(f"    Model successfully saved to {expected_model_path}")

    # =========================================================================
    # INFERENCE DEMONSTRATION
    # =========================================================================
    print("\n[5] Running Inference and Generating Submission...")

    # generate_submission handles loading the model, running inference on test set,
    # post-processing, and saving the CSV.
    generate_submission(cfg)

    expected_sub_path = os.path.join(cfg.submission_dir, "submission.csv")
    if not os.path.exists(expected_sub_path):
        raise FileNotFoundError(f"Submission file not found at {expected_sub_path}")

    # Validate submission format
    sub_df = pd.read_csv(expected_sub_path)
    if list(sub_df.columns) != ["id", "PredictionString"]:
        raise AssertionError(f"Invalid submission columns: {sub_df.columns}")
    if len(sub_df) == 0:
        raise AssertionError("Submission file is empty.")

    print(f"    Submission generated with {len(sub_df)} rows.")
    print("    First 3 rows:")
    print(sub_df.head(3))

    # =========================================================================
    # METRIC VERIFICATION
    # =========================================================================
    print("\n[6] Verifying Metric Function...")

    # Test Case 1: Identical strings
    score_1 = jaccard("hello world", "hello world")
    assert score_1 == 1.0, f"Expected 1.0, got {score_1}"

    # Test Case 2: Partial overlap
    # A = {hello, world}, B = {hello, python}
    # Intersection = {hello} (1), Union = {hello, world, python} (3) -> 1/3
    score_2 = jaccard("hello world", "hello python")
    expected_2 = 1.0 / 3.0
    assert abs(score_2 - expected_2) < 1e-6, f"Expected {expected_2}, got {score_2}"

    # Test Case 3: No overlap
    score_3 = jaccard("apple", "orange")
    assert score_3 == 0.0, f"Expected 0.0, got {score_3}"

    print("    Jaccard metric verification passed.")

    print("\n==== Demonstration Complete ====")


if __name__ == "__main__":
    main()
