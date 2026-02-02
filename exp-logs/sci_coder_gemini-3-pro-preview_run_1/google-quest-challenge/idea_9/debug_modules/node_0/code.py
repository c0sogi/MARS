import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.dataset import load_and_preprocess_data, StackExchangeDataset, CollateFn
from library.model import DualDistilRoBERTa
from library.engine import train_one_epoch, validate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("Starting demonstration...")

    # 1. Reproducibility
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Verify Metric Logic
    print("\nVerifying metric calculation...")
    # Create dummy ground truth and predictions
    # Shape: (10 samples, 30 targets)
    y_true_dummy = np.random.rand(10, 30)
    y_pred_dummy = np.random.rand(10, 30)

    # Ensure perfect correlation gives 1.0
    score_perfect = compute_metric(y_true_dummy, y_true_dummy)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Metric check failed: Expected 1.0, got {score_perfect}"

    # Ensure random correlation is valid float
    score_random = compute_metric(y_true_dummy, y_pred_dummy)
    assert (
        -1.0 <= score_random <= 1.0
    ), f"Metric check failed: Score {score_random} out of range"
    print("Metric verification passed.")

    # 3. Data Loading & Subsampling (for Speed)
    print("\nLoading and preprocessing data...")

    # Load dataframes
    train_df = load_and_preprocess_data("train", load_cached_data=False)
    val_df = load_and_preprocess_data("val", load_cached_data=False)
    test_df = load_and_preprocess_data("test", load_cached_data=False)

    # SUBSAMPLING: Take only 50 samples for this demo to ensure speed
    train_df = train_df.head(50).reset_index(drop=True)
    val_df = val_df.head(50).reset_index(drop=True)
    test_df = test_df.head(50).reset_index(drop=True)

    print(f"Subsampled Train shape: {train_df.shape}")
    print(f"Subsampled Val shape: {val_df.shape}")
    print(f"Subsampled Test shape: {test_df.shape}")

    # 4. Dataset & DataLoader
    print("\nInitializing Tokenizer and Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Use smaller max_len for demo speed
    demo_max_len = 128

    train_dataset = StackExchangeDataset(train_df, tokenizer, max_len=demo_max_len)
    val_dataset = StackExchangeDataset(val_df, tokenizer, max_len=demo_max_len)
    test_dataset = StackExchangeDataset(
        test_df, tokenizer, max_len=demo_max_len, is_test=True
    )

    collate_fn = CollateFn(tokenizer)

    # Use smaller batch size
    batch_size = 8

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 5. Model Initialization & Architecture Verification
    print("\nInitializing Model...")
    model = DualDistilRoBERTa()
    model.to(device)

    # Dry run with one batch to verify shapes
    print("Verifying model forward pass...")
    sample_batch = next(iter(train_loader))

    q_input_ids = sample_batch["q_input_ids"].to(device)
    q_mask = sample_batch["q_attention_mask"].to(device)
    a_input_ids = sample_batch["a_input_ids"].to(device)
    a_mask = sample_batch["a_attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(q_input_ids, q_mask, a_input_ids, a_mask)

    # Expected output shape: [Batch_Size, 30]
    assert outputs.shape == (
        q_input_ids.size(0),
        30,
    ), f"Model output shape mismatch. Expected {(q_input_ids.size(0), 30)}, got {outputs.shape}"
    print("Model forward pass verified.")

    # 6. Training Loop
    print("\nStarting Training Loop (1 Epoch)...")

    optimizer = AdamW(model.parameters(), lr=Config.LR_HEAD)

    # Scheduler setup
    num_training_steps = len(train_loader) * 1  # 1 epoch
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Train
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch=1
    )

    # Validate
    val_loss, val_score = validate(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Spearman: {val_score:.4f}")

    # 7. Inference & Submission
    print("\nGenerating Submission...")
    submission_path = "./working/demo_submission.csv"
    generate_submission(model, test_loader, device, output_path=submission_path)

    # 8. Verify Submission File
    print("Verifying submission file...")
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)

    # Check rows (should match subsampled test set size)
    assert len(sub_df) == len(
        test_df
    ), f"Submission row count mismatch. Expected {len(test_df)}, got {len(sub_df)}"

    # Check columns (qa_id + 30 targets)
    expected_cols = ["qa_id"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Check value range [0, 1]
    # Exclude qa_id
    preds = sub_df[Config.TARGET_COLS].values
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions are out of range [0, 1]."

    print("Submission verification passed.")
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
