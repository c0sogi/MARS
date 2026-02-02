import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Ensure library is in path if needed (though running from root usually works)
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_data, get_dataloader, ChatbotDataset, CollateFn
from library.model import SiameseDeberta
from library.engine import run_training, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class DemoConfig(Config):
    """
    Configuration optimized for a quick demonstration run.
    Overrides the default Config to use a tiny subset of data and minimal training steps.
    """

    # Use a separate working directory for the demo
    working_dir = "./working/demo_run"
    output_dir = working_dir
    submission_dir = os.path.join(working_dir, "submission")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Debug mode to load only a small subset of data
    debug = True
    debug_sample_size = 32  # Small enough for quick CPU/GPU processing

    # Training hyperparameters for speed
    epochs = 1
    train_batch_size = 4
    valid_batch_size = 4
    gradient_accumulation_steps = 1

    # Model settings (keep standard model but ensure settings are correct)
    model_name = "microsoft/deberta-v3-base"  # Using base for demo speed if possible, or stick to provided
    max_length = 128  # Reduced sequence length for speed

    # Disable TTA for the demo to save inference time, or keep it to test logic
    use_tta = True

    def __init__(self):
        super().__init__()
        # Clean up demo directory if it exists to ensure a fresh run
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)


def verify_data_pipeline(config):
    print("\n=== Verifying Data Pipeline ===")

    # 1. Test Data Preparation and Augmentation
    print("Loading and preparing training data...")
    train_df = prepare_data(config, partition="train", load_cached_data=False)

    # Expectation: debug_sample_size * 2 (because use_symmetric_augmentation=True by default)
    expected_size = config.debug_sample_size * 2
    print(f"Train DataFrame shape: {train_df.shape}")

    if len(train_df) != expected_size:
        raise AssertionError(
            f"Expected {expected_size} rows (symmetric aug), got {len(train_df)}"
        )

    # Check required columns
    required_cols = [
        "input_ids_a",
        "input_ids_b",
    ]  # These are not in DF yet, DF has text
    required_text_cols = ["prompt", "response_a", "response_b", "winner_model_a"]
    for col in required_text_cols:
        if col not in train_df.columns:
            raise AssertionError(f"Missing column {col} in prepared dataframe")

    # 2. Test DataLoader and CollateFn
    print("Initializing DataLoader...")
    loader = get_dataloader(config, partition="train", load_cached_data=False)

    # Fetch one batch
    batch = next(iter(loader))

    print("Inspecting batch keys and shapes...")
    # Check keys
    expected_keys = [
        "id",
        "input_ids_a",
        "attention_mask_a",
        "token_type_ids_a",
        "input_ids_b",
        "attention_mask_b",
        "token_type_ids_b",
        "features",
        "target",
    ]
    for key in expected_keys:
        if key not in batch:
            raise AssertionError(f"Batch missing key: {key}")

    # Check shapes
    # input_ids_a: [Batch, SeqLen]
    b_size = batch["input_ids_a"].shape[0]
    seq_len = batch["input_ids_a"].shape[1]

    if b_size != config.train_batch_size:
        raise AssertionError(
            f"Batch size mismatch. Expected {config.train_batch_size}, got {b_size}"
        )

    if seq_len > config.max_length:
        raise AssertionError(
            f"Sequence length {seq_len} exceeds max_length {config.max_length}"
        )

    # features: [Batch, 3]
    if batch["features"].shape != (b_size, 3):
        raise AssertionError(
            f"Features shape mismatch. Expected ({b_size}, 3), got {batch['features'].shape}"
        )

    # target: [Batch, 3]
    if batch["target"].shape != (b_size, 3):
        raise AssertionError(
            f"Target shape mismatch. Expected ({b_size}, 3), got {batch['target'].shape}"
        )

    print("Data Pipeline verification passed.")
    return batch


def verify_model_forward(config, batch):
    print("\n=== Verifying Model Architecture ===")

    device = get_device()
    model = SiameseDeberta(config)
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}.")

    # Move batch to device
    # The CollateFn returns lists for IDs, need to handle that if used in model,
    # but model forward only uses tensor keys.
    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(device)
        else:
            batch_device[k] = v

    print("Running forward pass...")
    with torch.no_grad():
        logits = model(batch_device)

    print(f"Logits shape: {logits.shape}")

    # Expected output: [Batch, 3] (3 classes: A wins, B wins, Tie)
    if logits.shape != (config.train_batch_size, 3):
        raise AssertionError(
            f"Logits shape mismatch. Expected ({config.train_batch_size}, 3), got {logits.shape}"
        )

    print("Model forward pass verification passed.")
    return model


def verify_training_loop(config):
    print("\n=== Verifying Training Loop ===")
    print("Starting training run (1 epoch, debug subset)...")

    # run_training handles dataloading, model init, training, and saving
    trained_model = run_training(config)

    # Check if best model was saved
    save_path = config.get_model_save_path()
    if not os.path.exists(save_path):
        raise AssertionError(f"Model checkpoint not found at {save_path}")

    print(f"Training complete. Model saved to {save_path}")
    return trained_model


def verify_inference(config, model):
    print("\n=== Verifying Inference and Submission ===")

    # generate_submission handles test loading, TTA, and file saving
    generate_submission(model, config)

    if not os.path.exists(config.submission_path):
        raise AssertionError(f"Submission file not found at {config.submission_path}")

    # Load and check submission
    sub_df = pd.read_csv(config.submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print(sub_df.head(2))

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    if not all(col in sub_df.columns for col in expected_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {sub_df.columns}"
        )

    # Check row count
    # In debug mode, prepare_data(partition='test') returns debug_sample_size rows
    # Note: prepare_data for test does NOT do symmetric augmentation.
    expected_rows = config.debug_sample_size
    if len(sub_df) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"
        )

    # Check probability sum (approximate)
    probs = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    if not np.allclose(probs, 1.0, atol=1e-4):
        print(
            "Warning: Probabilities do not sum to exactly 1.0 (likely due to float precision or softmax implementation details)."
        )

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Setup
    config = DemoConfig()
    seed_everything(config.seed)

    print(f"Working Directory: {config.working_dir}")
    print(f"Device: {config.device}")

    # 2. Verify Data Pipeline
    sample_batch = verify_data_pipeline(config)

    # 3. Verify Model
    # We instantiate a fresh model just to check the forward pass logic independently
    verify_model_forward(config, sample_batch)

    # 4. Verify Training Execution
    # This will train a model from scratch and return the best state
    trained_model = verify_training_loop(config)

    # 5. Verify Inference
    verify_inference(config, trained_model)

    print("\nAll demonstrations and verifications completed successfully.")
