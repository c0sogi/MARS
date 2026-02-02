import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import load_data, QuestDataset, CollateFactory, get_tokenizer
from library.model import QuestModel
from library.engine import run_training, predict_and_submit


def verify_metric():
    print("\n--- Verifying Metric ---")
    # Create dummy data: perfect correlation
    y_true = np.random.rand(10, 30)
    y_pred = y_true.copy()

    score = compute_spearman_metric(y_true, y_pred)
    print(f"Perfect correlation score: {score}")
    assert np.isclose(score, 1.0), "Metric should be 1.0 for identical arrays"

    # Create dummy data: random correlation
    y_pred_random = np.random.rand(10, 30)
    score_random = compute_spearman_metric(y_true, y_pred_random)
    print(f"Random correlation score: {score_random}")
    assert -1.0 <= score_random <= 1.0, "Metric must be between -1 and 1"
    print("Metric verification passed.")


def verify_data_pipeline():
    print("\n--- Verifying Data Pipeline ---")
    # Load data in debug mode
    train_df, val_df, test_df = load_data(load_cached_data=False, debug=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    assert not train_df.empty, "Train DataFrame is empty"
    assert not val_df.empty, "Val DataFrame is empty"
    assert not test_df.empty, "Test DataFrame is empty"

    # Check Dataset
    ds = QuestDataset(train_df, is_test=False)
    sample = ds[0]
    print(f"Sample keys: {sample.keys()}")
    assert "question" in sample
    assert "answer" in sample
    assert "labels" in sample
    assert sample["labels"].shape[0] == 30, "Labels dimension mismatch"

    # Check Collator
    tokenizer = get_tokenizer()
    collate_fn = CollateFactory(tokenizer)

    # Create a small batch
    batch_samples = [ds[i] for i in range(4)]
    batch = collate_fn(batch_samples)

    print(f"Batch keys: {batch.keys()}")
    assert "q_input_ids" in batch
    assert "a_input_ids" in batch
    assert "labels" in batch

    # Check shapes
    batch_size = 4
    assert batch["q_input_ids"].shape[0] == batch_size
    assert batch["a_input_ids"].shape[0] == batch_size
    assert batch["labels"].shape == (batch_size, 30)

    print("Data pipeline verification passed.")
    return batch


def verify_model(batch):
    print("\n--- Verifying Model Architecture ---")
    device = Config.device
    model = QuestModel()
    model.to(device)
    model.eval()

    # Move batch to device
    q_ids = batch["q_input_ids"].to(device)
    q_mask = batch["q_attention_mask"].to(device)
    a_ids = batch["a_input_ids"].to(device)
    a_mask = batch["a_attention_mask"].to(device)

    with torch.no_grad():
        logits = model(q_ids, q_mask, a_ids, a_mask)

    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (4, 30), f"Expected shape (4, 30), got {logits.shape}"
    print("Model verification passed.")


def run_demo_training_and_inference():
    print("\n--- Running Demo Training & Inference ---")

    # Run training
    # This uses the modified Config settings
    run_training()

    # Check if model was saved
    assert os.path.exists(Config.model_save_path), "Model file was not saved."
    print("Training completed and model saved.")

    # Run inference
    predict_and_submit()

    # Check submission
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Verify submission format
    # Should have qa_id + 30 target columns = 31 columns
    assert sub_df.shape[1] == 31, f"Expected 31 columns, got {sub_df.shape[1]}"
    assert "qa_id" in sub_df.columns, "qa_id column missing"

    # Verify values are probabilities
    target_cols = [c for c in sub_df.columns if c != "qa_id"]
    assert len(target_cols) == 30

    vals = sub_df[target_cols].values
    assert vals.min() >= 0.0 and vals.max() <= 1.0, "Predictions out of range [0, 1]"

    print("Demo run completed successfully.")


if __name__ == "__main__":
    # 1. Modify Config for Speed/Demo
    print("Configuring environment for demo run...")
    Config.debug = True
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.gradient_accumulation_steps = 1
    Config.working_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_dir = "./working/demo_run"  # Save submission here for demo
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds
    seed_everything(Config.seed)

    # 2. Verify Metric Logic
    verify_metric()

    # 3. Verify Data Pipeline
    batch = verify_data_pipeline()

    # 4. Verify Model
    verify_model(batch)

    # 5. Run Training and Inference
    run_demo_training_and_inference()
