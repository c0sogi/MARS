import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import logging

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_score
from library.data import load_and_process_data, ChatbotDataset, get_dataloaders
from library.modeling import SiameseHybridModel
from library.engine import train_model, predict

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # 1. Setup and Configuration
    # We modify the Config class directly to optimize for a fast demonstration run.
    print("\n[1] Configuring environment for fast demo...")
    set_seed(42)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 rows
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run

    # Ensure working directory exists (usually handled by Config.setup, but good to be explicit)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading and Dataset Verification
    print("\n[2] Verifying Data Loading and Dataset Logic...")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load raw data (internally handles caching)
    train_df = load_and_process_data("train", load_cached_data=False)

    # Take a small slice for manual dataset verification
    demo_df = train_df.head(4).copy()

    # Instantiate Dataset
    dataset = ChatbotDataset(demo_df, tokenizer, max_length=128, is_test=False)

    # Retrieve one sample
    sample = dataset[0]

    # Assertions to verify Dataset logic
    print("   Verifying dataset sample keys and shapes...")
    expected_keys = {
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalars",
        "labels",
    }
    assert (
        set(sample.keys()) == expected_keys
    ), f"Missing keys in dataset sample. Found: {sample.keys()}"

    # Check tensor shapes
    # Input IDs should be 1D tensors of length max_length (due to padding)
    assert sample["input_ids_a"].shape == (
        128,
    ), f"Incorrect shape for input_ids_a: {sample['input_ids_a'].shape}"
    assert sample["scalars"].shape == (
        3,
    ), f"Incorrect shape for scalars: {sample['scalars'].shape}"
    assert sample["labels"].shape == (
        3,
    ), f"Incorrect shape for labels: {sample['labels'].shape}"

    print("   Dataset verification passed.")

    # 3. Model Instantiation and Forward Pass
    print("\n[3] Verifying Model Architecture...")

    model = SiameseHybridModel(model_name=Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()

    # Create a batch from the sample (add batch dimension)
    batch = {k: v.unsqueeze(0).to(Config.DEVICE) for k, v in sample.items()}

    # Run Forward Pass
    with torch.no_grad():
        # Note: The model forward signature requires specific arguments
        logits = model(
            input_ids_a=batch["input_ids_a"],
            attention_mask_a=batch["attention_mask_a"],
            input_ids_b=batch["input_ids_b"],
            attention_mask_b=batch["attention_mask_b"],
            scalars=batch["scalars"],
        )

    # Check output
    assert logits.shape == (
        1,
        3,
    ), f"Model output shape mismatch. Expected (1, 3), got {logits.shape}"
    print("   Model forward pass successful. Output shape verified.")

    # 4. Full Training Loop Execution
    print("\n[4] Executing Training Loop (Engine)...")

    # train_model handles dataloaders, optimization, and saving internally
    trained_model = train_model(tokenizer)

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH}"
    print(f"   Training complete. Model saved to {Config.MODEL_PATH}")

    # 5. Inference / Prediction
    print("\n[5] Executing Inference (Engine)...")

    # predict handles test dataloader creation and submission file generation
    predict(tokenizer)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check if probabilities sum roughly to 1 (allow small float error)
    row_sums = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    # In a real scenario, softmax ensures sum is 1.0. We check if it's close.
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print(f"   Inference complete. Submission generated with shape {sub_df.shape}")

    # 6. Metric Utility Verification
    print("\n[6] Verifying Metric Calculation...")

    # Create synthetic ground truth and predictions
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])  # Soft probs

    metrics = compute_score(y_true, y_pred)

    assert "log_loss" in metrics
    assert "accuracy" in metrics
    assert (
        metrics["accuracy"] == 1.0
    ), "Accuracy calculation incorrect for perfect match indices"

    print(f"   Metrics verified: {metrics}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
