import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, jaccard, get_average_jaccard, clean_text
from library.data import prepare_train_features, prepare_test_features, QADataset
from library.model import CustomXLMRoberta
from library.train import train_seed
from library.inference import predict_and_ensemble


def create_demo_data(base_dir):
    """
    Creates a tiny subset of the data for demonstration purposes to ensure
    the code runs quickly.
    """
    print(f"Creating demo data in {base_dir}...")
    os.makedirs(base_dir, exist_ok=True)

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample top 10 rows for train/val and 5 for test
    demo_train = orig_train.head(10).copy()
    demo_val = orig_val.head(5).copy()
    demo_test = orig_test.head(5).copy()

    # Save to demo directory
    train_path = os.path.join(base_dir, "train.csv")
    val_path = os.path.join(base_dir, "val.csv")
    test_path = os.path.join(base_dir, "test.csv")

    demo_train.to_csv(train_path, index=False)
    demo_val.to_csv(val_path, index=False)
    demo_test.to_csv(test_path, index=False)

    # Create sample submission for the demo test set
    sub_path = os.path.join(base_dir, "sample_submission.csv")
    demo_sub = pd.DataFrame(
        {"id": demo_test["id"], "PredictionString": [""] * len(demo_test)}
    )
    demo_sub.to_csv(sub_path, index=False)

    return train_path, val_path, test_path, sub_path


def patch_config(demo_paths):
    """
    Overrides Config parameters to use demo data and run fast.
    """
    train_path, val_path, test_path, sub_path = demo_paths

    # Paths
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path
    Config.SAMPLE_SUBMISSION = sub_path

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.SEEDS = [42]  # Only run one seed
    Config.TRAIN_BATCH_SIZE = 2
    Config.VALID_BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script

    # Re-run setup to create new directories
    Config.setup()

    print("Config patched for demo run.")


def test_utils():
    print("\n=== Testing Utils ===")

    # Test Jaccard
    s1 = "This is a test"
    s2 = "This is a test string"
    score = jaccard(s1, s2)
    # intersection: {this, is, a, test} (4)
    # union: {this, is, a, test, string} (5)
    # score: 4/5 = 0.8
    assert (
        abs(score - 0.8) < 1e-6
    ), f"Jaccard calculation failed. Expected 0.8, got {score}"
    print(f"Jaccard check passed: {score}")

    # Test Cleaning
    raw = "  Text   with   spaces  "
    cleaned = clean_text(raw)
    assert cleaned == "Text with spaces", "Text cleaning failed"
    print("Text cleaning check passed.")


def test_data_pipeline():
    print("\n=== Testing Data Pipeline ===")

    # Run data preparation
    # load_cached_data=False ensures we actually run the processing logic
    dataset = prepare_train_features(seed=42, load_cached_data=False)

    print(f"Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty"

    # Check one item
    item = dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "relevance_labels",
    ]
    for key in required_keys:
        assert key in item, f"Missing key in dataset item: {key}"
        assert isinstance(item[key], torch.Tensor), f"{key} is not a tensor"

    print("Dataset item keys and types verified.")
    return dataset


def test_model_forward(dataset):
    print("\n=== Testing Model Architecture ===")

    device = Config.DEVICE
    model = CustomXLMRoberta(Config.MODEL_NAME)
    model.to(device)
    model.eval()

    # Create a small batch
    batch_size = 2
    input_ids = torch.stack([dataset[i]["input_ids"] for i in range(batch_size)]).to(
        device
    )
    attention_mask = torch.stack(
        [dataset[i]["attention_mask"] for i in range(batch_size)]
    ).to(device)

    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        start_logits, end_logits, rel_logits = model(input_ids, attention_mask)

    # Verify shapes
    seq_len = Config.MAX_LEN
    assert start_logits.shape == (
        batch_size,
        seq_len,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        batch_size,
        seq_len,
    ), f"End logits shape mismatch: {end_logits.shape}"
    assert rel_logits.shape == (
        batch_size,
    ), f"Relevance logits shape mismatch: {rel_logits.shape}"

    print("Model forward pass shapes verified.")

    # Cleanup
    del model
    torch.cuda.empty_cache()


def test_training_loop():
    print("\n=== Testing Training Loop ===")

    # This calls library.train.train_seed
    # It will train for 1 epoch on the demo data and save the checkpoint
    train_seed(42)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_seed_42.bin")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Training completed. Checkpoint saved at {checkpoint_path}")


def test_inference_pipeline():
    print("\n=== Testing Inference Pipeline ===")

    # This calls library.inference.predict_and_ensemble
    # It loads the checkpoint we just trained and generates a submission
    predict_and_ensemble()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df)}")

    # Verify format
    assert (
        "id" in df.columns and "PredictionString" in df.columns
    ), "Submission columns mismatch"

    # Check against sample submission
    sample = pd.read_csv(Config.SAMPLE_SUBMISSION)
    assert len(df) == len(sample), "Submission length mismatch with sample"

    print("Inference pipeline verified successfully.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    demo_data_dir = "./working/demo_run/demo_data"

    # 2. Create Data & Patch Config
    demo_paths = create_demo_data(demo_data_dir)
    patch_config(demo_paths)

    # 3. Run Tests
    test_utils()
    dataset = test_data_pipeline()
    test_model_forward(dataset)
    test_training_loop()
    test_inference_pipeline()

    print("\nAll demonstrations completed successfully.")
