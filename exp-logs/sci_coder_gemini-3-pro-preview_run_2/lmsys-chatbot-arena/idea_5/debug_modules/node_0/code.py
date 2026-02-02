import os
import sys
import shutil
import logging
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import load_data, ChatbotDataset, get_dataloaders
from library.model import SiameseModel
from library.trainer import run_fold
from library.inference import generate_submission

# Suppress library logging to keep output clean
logging.getLogger("data").setLevel(logging.WARNING)
logging.getLogger("trainer").setLevel(logging.WARNING)
logging.getLogger("inference").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)


def run_demo():
    print("Starting Demo Execution...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1/6] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.EPOCHS = 1
    Config.N_FOLDS = 1  # Only run one fold
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect paths to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated. Output directory:", Config.WORKING_DIR)

    # ==========================================
    # 2. Data Loading & Subsetting
    # ==========================================
    print("\n[2/6] Loading and subsetting data...")

    # Load full data first (ignoring cache to ensure we start fresh)
    full_train_df, full_test_df = load_data(load_cached_data=False)

    # Create tiny subsets for demonstration
    # 20 samples for training, 10 for testing
    subset_train_df = full_train_df.head(20).copy()
    subset_test_df = full_test_df.head(10).copy()

    # Save subsets to cache.
    # Important: The library's `load_data` checks these paths. By saving our subsets here,
    # `generate_submission` (which calls `load_data`) will use our small datasets.
    subset_train_df.to_parquet(
        os.path.join(Config.CACHE_DIR, "train_processed.parquet"), index=False
    )
    subset_test_df.to_parquet(
        os.path.join(Config.CACHE_DIR, "test_processed.parquet"), index=False
    )

    print(f"Subset created: Train={len(subset_train_df)}, Test={len(subset_test_df)}")

    # ==========================================
    # 3. Dataset & DataLoader Verification
    # ==========================================
    print("\n[3/6] Verifying Dataset and DataLoader...")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Instantiate dataset
    dataset = ChatbotDataset(subset_train_df, tokenizer, max_length=128, is_test=False)

    # Check single item
    item = dataset[0]
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "meta_features",
        "target",
    ]

    for key in required_keys:
        assert key in item, f"Missing key {key} in dataset item"
        assert isinstance(item[key], torch.Tensor), f"Item {key} is not a tensor"

    print("Dataset item keys and types verified.")
    print(f"Input IDs shape: {item['input_ids_a'].shape}")
    print(f"Target shape: {item['target'].shape}")

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print("\n[4/6] Verifying SiameseModel...")

    device = torch.device("cpu")  # Use CPU for quick logic check
    model = SiameseModel()
    model.to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    dummy_batch = {
        "input_ids_a": item["input_ids_a"]
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device),
        "attention_mask_a": item["attention_mask_a"]
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device),
        "input_ids_b": item["input_ids_b"]
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device),
        "attention_mask_b": item["attention_mask_b"]
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device),
        "meta_features": item["meta_features"]
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device),
    }

    with torch.no_grad():
        outputs = model(
            dummy_batch["input_ids_a"],
            dummy_batch["attention_mask_a"],
            dummy_batch["input_ids_b"],
            dummy_batch["attention_mask_b"],
            dummy_batch["meta_features"],
        )

    assert outputs.shape == (
        batch_size,
        3,
    ), f"Expected output shape ({batch_size}, 3), got {outputs.shape}"
    print("Model forward pass successful. Output shape verified.")

    # ==========================================
    # 5. Training Simulation (Fold 0)
    # ==========================================
    print("\n[5/6] Running training for Fold 0...")

    # Split subset into train/val for the fold
    # Since we have 20 samples, let's use 16 for train, 4 for val
    fold_train_df = subset_train_df.iloc[:16].reset_index(drop=True)
    fold_val_df = subset_train_df.iloc[16:].reset_index(drop=True)

    # Run training
    # This will save 'best_model_fold_0.pth' in Config.MODEL_OUTPUT_DIR
    best_loss = run_fold(0, fold_train_df, fold_val_df, tokenizer)

    expected_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "best_model_fold_0.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {expected_model_path}")

    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {expected_model_path}")

    # ==========================================
    # 6. Inference Simulation
    # ==========================================
    print("\n[6/6] Generating submission...")

    # generate_submission uses Config.N_FOLDS. We set it to 1, so it will look for fold 0 model.
    # It also calls load_data(load_cached_data=True), which will pick up our cached subset_test_df.
    generate_submission(load_cached_data=True)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == len(
        subset_test_df
    ), f"Submission length mismatch. Expected {len(subset_test_df)}, got {len(sub_df)}"
    assert all(
        col in sub_df.columns
        for col in ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    ), "Missing columns in submission"

    print("Submission generated and verified.")
    print(sub_df.head().to_string())

    print("\nDemo execution completed successfully!")


if __name__ == "__main__":
    run_demo()
