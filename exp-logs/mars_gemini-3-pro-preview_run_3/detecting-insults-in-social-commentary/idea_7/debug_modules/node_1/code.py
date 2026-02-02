import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil
import logging

# Ensure library modules can be imported
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, decode_text, get_logger
from library.dataset import load_processed_data, InsultDataset, get_dataloader
from library.model import InsultModel
from library.awp import AWP
from library.train import run_training
from library.inference import run_inference
from transformers import AutoTokenizer


def main():
    print("=== Starting Demonstration of Insult Detection Pipeline ===")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")
    # We override Config attributes to use a tiny model and minimal data/epochs
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample for training demo
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.GRAD_ACCUM_STEPS = 1
    Config.MODEL_NAME = "prajjwal1/bert-tiny"  # Lightweight model for speed
    Config.USE_AWP = True
    Config.AWP_START_EPOCH = 0  # Activate AWP immediately
    Config.SEEDS = [42]  # Run only one seed

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Model: {Config.MODEL_NAME}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # Test Seeding
    seed_everything(42)

    # Test Text Decoding
    raw_text = "Hello\\nWorld \\xe2\\x80\\x9cQuote\\xe2\\x80\\x9d"
    decoded = decode_text(raw_text)
    print(f"    Raw: {raw_text}")
    print(f"    Decoded: {decoded}")
    assert "“" in decoded, "Text decoding failed to handle unicode escape."
    assert "\n" in decoded, "Text decoding failed to handle newlines."

    # Test Logger
    logger = get_logger("demo")
    logger.info("    Logger initialized successfully.")

    # ==========================================
    # 3. Verify Dataset & DataLoader
    # ==========================================
    print("\n[3] Verifying Dataset and DataLoader...")

    # Load processed data (Debug mode)
    df_train = load_processed_data(Config.TRAIN_PATH, "train_demo.parquet", debug=True)
    assert len(df_train) == Config.DEBUG_SAMPLE_SIZE, "Debug sample size mismatch."
    assert "Comment" in df_train.columns, "Missing 'Comment' column."

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Dataset
    dataset = InsultDataset(df_train, tokenizer, max_len=32)
    sample_item = dataset[0]

    # Check Item Structure
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "target" in sample_item
    assert torch.is_tensor(sample_item["input_ids"])
    print("    Dataset item structure verified.")

    # Create DataLoader
    dataloader = get_dataloader(
        df_train, tokenizer, batch_size=Config.TRAIN_BATCH_SIZE, max_len=32
    )
    batch = next(iter(dataloader))

    # Check Batch Structure
    assert batch["input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    assert batch["input_ids"].shape[1] == 32  # max_len
    print("    DataLoader batch shapes verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InsultModel(Config.MODEL_NAME)
    model.to(device)
    model.train()

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    targets = batch["target"].to(device).unsqueeze(1)

    # Forward Pass
    outputs = model(input_ids, mask)

    assert outputs.shape == (
        Config.TRAIN_BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {outputs.shape}"
    print("    Forward pass successful. Output shape verified.")

    # ==========================================
    # 5. Verify Adversarial Weight Perturbation (AWP)
    # ==========================================
    print("\n[5] Verifying AWP Logic...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Initialize AWP with high learning rate to ensure visible perturbation
    awp = AWP(model, optimizer, adv_lr=0.5, adv_eps=1.0, start_epoch=0)

    # 1. Compute Gradients
    loss = criterion(outputs, targets)
    loss.backward()

    # 2. Identify a weight parameter to track
    target_param = None
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and "weight" in name:
            target_param = param
            break

    if target_param is not None:
        original_weights = target_param.data.clone()

        # 3. Attack (Perturb Weights)
        awp.attack()
        assert not torch.equal(
            target_param.data, original_weights
        ), "AWP Attack did not change weights."
        print("    AWP Attack: Weights perturbed.")

        # 4. Restore (Revert Weights)
        awp.restore()
        assert torch.allclose(
            target_param.data, original_weights
        ), "AWP Restore did not revert weights."
        print("    AWP Restore: Weights reverted.")
    else:
        print("    Warning: No suitable parameter found to verify AWP.")

    # ==========================================
    # 6. Run Training Pipeline
    # ==========================================
    print("\n[6] Running Full Training Pipeline (Seed 42)...")

    # This calls library.train.run_training
    # It will use the Config overrides we set at the beginning
    try:
        run_training(seed=42)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify Model Artifact
    model_path = os.path.join(Config.OUTPUT_DIR, "model_seed_42.bin")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print(f"    Training completed. Model saved to {model_path}")

    # ==========================================
    # 7. Run Inference Pipeline
    # ==========================================
    print("\n[7] Running Inference Pipeline...")

    # This calls library.inference.run_inference
    # Note: run_inference forces debug=False for test data loading in the library code.
    # Since we use bert-tiny, inference on the full test set (2647 rows) will still be fast.
    try:
        run_inference()
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"    Submission shape: {sub_df.shape}")

    # Expected shape: (2647, 3) or (2647, 2) depending on if sample_submission_null has columns.
    # Metadata says sample_submission_null has 2647 rows.
    assert (
        len(sub_df) == 2647
    ), f"Submission length mismatch. Expected 2647, got {len(sub_df)}"
    assert "Insult" in sub_df.columns, "Submission missing 'Insult' column."

    # Check prediction range
    preds = sub_df["Insult"].values
    assert preds.min() >= 0.0 and preds.max() <= 1.0, "Predictions out of range [0, 1]."

    print("    Inference completed successfully.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
