import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
import torch
import transformers

# Suppress warnings and verbose logs
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_processing import FeatureEngineer, get_dataloaders
from library.model import SiameseDeberta
from library.train import run_training
from library.inference import generate_predictions


def main():
    print("======================================================")
    print("      LIBRARY DEMONSTRATION & VERIFICATION SCRIPT     ")
    print("======================================================")

    # 1. Setup & Configuration Override
    # We override the Config to run a fast, lightweight demo.
    print("\n[1] Configuring environment for fast demo execution...")

    seed_everything(42)

    # Modify Config for speed and isolation
    Config.debug = True  # Triggers subsampling in get_dataloaders
    Config.exp_name = "demo_verification"
    Config.epochs = 1
    Config.train_batch_size = 2
    Config.valid_batch_size = 2
    Config.gradient_accumulation_steps = 1
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Update paths based on new exp_name
    Config.working_dir = f"./working/{Config.exp_name}/"
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)

    # Create directories (Config.setup() is usually called inside get_dataloaders,
    # but we call it here to ensure dirs exist for manual checks)
    Config.setup()

    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Device: {Config.device}")

    # 2. Verify Feature Engineering
    print("\n[2] Verifying FeatureEngineer...")

    # Create dummy data
    dummy_data = pd.DataFrame(
        {
            "prompt": ["Help me"],
            "response_a": ["Short response"],
            "response_b": ["This is a much longer response than A"],
        }
    )

    fe = FeatureEngineer()
    features = fe.extract_features(dummy_data)

    # Expected: 1 sample, 6 features
    assert features.shape == (
        1,
        6,
    ), f"Feature shape mismatch. Expected (1, 6), got {features.shape}"

    # Check logic: response_b is longer, so char_diff (A - B) should be negative
    # Feature 0 is char diff
    char_diff = features[0, 0]
    assert (
        char_diff < 0
    ), f"Feature logic error: Expected negative char diff, got {char_diff}"

    print("    FeatureEngineer verification passed.")

    # 3. Verify Data Processing & Loading
    print("\n[3] Verifying Data Loading (get_dataloaders)...")

    # Force reload to ensure processing logic runs
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Check keys
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalar_features",
        "labels",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Check shapes
    # Batch size should be Config.train_batch_size (2)
    b_size = batch["input_ids_a"].size(0)
    assert (
        b_size == Config.train_batch_size
    ), f"Batch size mismatch. Expected {Config.train_batch_size}, got {b_size}"

    # Labels should be (batch, 3)
    assert batch["labels"].shape == (
        b_size,
        3,
    ), f"Labels shape mismatch. Expected ({b_size}, 3), got {batch['labels'].shape}"

    # Scalar features should be (batch, 6)
    assert batch["scalar_features"].shape == (
        b_size,
        6,
    ), f"Scalar features shape mismatch."

    print("    Data Loading verification passed.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying SiameseDeberta Model...")

    model = SiameseDeberta()
    model.to(Config.device)
    model.eval()

    # Move batch to device
    inputs = {
        "input_ids_a": batch["input_ids_a"].to(Config.device),
        "attention_mask_a": batch["attention_mask_a"].to(Config.device),
        "input_ids_b": batch["input_ids_b"].to(Config.device),
        "attention_mask_b": batch["attention_mask_b"].to(Config.device),
        "scalar_features": batch["scalar_features"].to(Config.device),
    }

    with torch.no_grad():
        logits = model(**inputs)

    # Check output shape: (batch, 3)
    assert logits.shape == (
        b_size,
        3,
    ), f"Model output shape mismatch. Expected ({b_size}, 3), got {logits.shape}"

    # Check for NaNs
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    print("    Model architecture verification passed.")

    # 5. Verify Training Loop
    print("\n[5] Running Training Loop (run_training)...")

    # This will use the debug loaders and run for 1 epoch
    try:
        run_training()
    except Exception as e:
        print(f"    Training failed with error: {e}")
        raise e

    # Check if model was saved
    assert os.path.exists(
        Config.model_save_path
    ), f"Model file not found at {Config.model_save_path}"
    print(f"    Training complete. Model saved to {Config.model_save_path}")

    # 6. Verify Inference Loop
    print("\n[6] Running Inference Loop (generate_predictions)...")

    try:
        generate_predictions(load_cached_data=True)
    except Exception as e:
        print(f"    Inference failed with error: {e}")
        raise e

    # Check if submission file exists
    assert os.path.exists(
        Config.submission_path
    ), f"Submission file not found at {Config.submission_path}"

    # Validate Submission Content
    sub_df = pd.read_csv(Config.submission_path)
    print(f"    Submission loaded. Shape: {sub_df.shape}")

    required_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert all(
        col in sub_df.columns for col in required_cols
    ), f"Submission missing columns. Found: {sub_df.columns}"

    # In debug mode, we subsampled test_df to 50 rows in data_processing.py
    # get_dataloaders handles subsampling. inference.py loads test.csv again.
    # inference.py logic: "if Config.debug: test_df = test_df.head(len(all_probs))"
    # So the shapes should align.
    assert len(sub_df) > 0, "Submission file is empty."

    # Check probability sum (approx 1.0)
    prob_sum = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    # Softmax ensures sum is 1.0. Allow small float error.
    assert np.allclose(prob_sum, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("    Inference verification passed.")

    print("\n======================================================")
    print("      ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY  ")
    print("======================================================")


if __name__ == "__main__":
    main()
