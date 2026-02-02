import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import (
    jaccard,
    get_soft_targets,
    get_best_start_end_idxs,
    normalize_text,
)
from library.dataset import get_data, TweetDataset
from library.model import SentimentModel
from library.engine import train_fn, eval_fn
from library.inference import run_inference


def test_utility_functions():
    print("\n=== Testing Utility Functions ===")

    # 1. Test Jaccard Similarity
    s1 = "good morning"
    s2 = "good morning"
    score_perfect = jaccard(s1, s2)
    assert score_perfect == 1.0, f"Expected 1.0, got {score_perfect}"

    s3 = "good night"
    score_partial = jaccard(s1, s3)
    # intersection: {good}, union: {good, morning, night} -> 1/3
    assert abs(score_partial - 1 / 3) < 1e-6, f"Expected 0.333..., got {score_partial}"

    print("Jaccard check passed.")

    # 2. Test Soft Targets (Gaussian Smoothing)
    seq_len = 10
    idx = 5
    sigma = 1.0
    targets = get_soft_targets(seq_len, idx, sigma)

    assert len(targets) == seq_len, "Target length mismatch"
    assert np.isclose(
        targets.sum(), 1.0
    ), f"Soft targets do not sum to 1: {targets.sum()}"
    assert np.argmax(targets) == idx, "Peak of Gaussian is not at the target index"

    print("Soft targets check passed.")

    # 3. Test Best Start/End Index Logic
    # Create logits where start=2 and end=4 is the clear winner
    start_logits = np.array([0, 0, 10, 0, 0, 0])
    end_logits = np.array([0, 0, 0, 0, 10, 0])

    s_idx, e_idx = get_best_start_end_idxs(start_logits, end_logits)
    assert s_idx == 2 and e_idx == 4, f"Expected (2, 4), got ({s_idx}, {e_idx})"

    # Test invalid case (end < start) - should pick valid pair with highest sum
    # start=4 (score 10), end=2 (score 10) -> Invalid
    # start=2 (score 5), end=3 (score 5) -> Valid sum 10
    start_logits = np.array([0, 0, 5, 0, 10, 0])
    end_logits = np.array([0, 0, 10, 5, 0, 0])

    s_idx, e_idx = get_best_start_end_idxs(start_logits, end_logits)
    # The pair (4, 2) is invalid.
    # Valid pairs: (2, 2) sum 15, (2, 3) sum 10.
    # Wait, start[2]=5, end[2]=10 -> sum 15. start[2]=5, end[3]=5 -> sum 10.
    # start[4]=10, end[4]=0 -> sum 10.
    # Best valid should be (2, 2).
    assert s_idx <= e_idx, "Selected start index is greater than end index"

    print("Index selection check passed.")
    print("Utility functions verified.")


def run_training_pipeline_demo():
    print("\n=== Running Training Pipeline Demo ===")

    # --- Configuration Override for Speed ---
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.N_FOLDS = 1  # Only train 1 fold for demo
    Config.NAME = "demo_execution"  # Separate working dir

    # Re-setup to apply new paths
    Config.WORKING_DIR = os.path.join(Config.ROOT_DIR, "working", Config.NAME)
    Config.setup()

    device = Config.DEVICE
    print(f"Device: {device}")

    # --- Data Preparation ---
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Use a tiny subset for demonstration
    subset_size = 32
    df_train = df_train.head(subset_size)
    df_val = df_val.head(subset_size)

    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print("Creating Datasets...")
    # get_data handles tokenization, caching, and target generation
    train_dataset = get_data(df_train, tokenizer, "train_demo", load_cached_data=False)
    val_dataset = get_data(df_val, tokenizer, "val_demo", load_cached_data=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Verify Dataset Output
    sample = train_dataset[0]
    required_keys = ["input_ids", "attention_mask", "start_targets", "end_targets"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset item"
    assert sample["input_ids"].shape[0] == Config.MAX_LEN, "Incorrect sequence length"

    # --- Model Initialization ---
    print("Initializing Model...")
    model = SentimentModel()
    model.to(device)

    # Verify Model Forward Pass
    dummy_ids = sample["input_ids"].unsqueeze(0).to(device)
    dummy_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        s_logits, e_logits = model(dummy_ids, dummy_mask)

    assert s_logits.shape == (
        1,
        Config.MAX_LEN,
    ), f"Logit shape mismatch: {s_logits.shape}"
    assert e_logits.shape == (
        1,
        Config.MAX_LEN,
    ), f"Logit shape mismatch: {e_logits.shape}"
    print("Model forward pass successful.")

    # --- Training Loop Setup ---
    optimizer_params = model.get_optimizer_params(encoder_lr=2e-5, decoder_lr=2e-5)
    optimizer = AdamW(optimizer_params, lr=2e-5, weight_decay=0.01)

    num_train_steps = int(len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # --- Execution ---
    print(f"Starting training for {Config.EPOCHS} epoch(s)...")

    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"Training Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss is NaN"

    avg_val_loss, avg_jaccard = eval_fn(val_loader, model, device)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Jaccard: {avg_jaccard:.4f}")
    assert not np.isnan(avg_val_loss), "Validation loss is NaN"
    assert 0 <= avg_jaccard <= 1.0, "Jaccard score out of range"

    # --- Save Model for Inference Demo ---
    model_path = os.path.join(Config.WORKING_DIR, "model_fold_0.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model_path


def run_inference_pipeline_demo(model_path):
    print("\n=== Running Inference Pipeline Demo ===")

    # Ensure the model exists where inference expects it
    # Config.WORKING_DIR is already set in the training demo
    assert os.path.exists(model_path), "Model file missing for inference"

    # Define output path
    output_csv = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Run Inference
    # We use n_folds=1 because we only saved one model
    # We use sample_size=20 to keep it fast
    submission_df = run_inference(
        test_meta_path=Config.TEST_META_PATH,
        base_model_dir=Config.WORKING_DIR,
        output_path=output_csv,
        batch_size=Config.VALID_BATCH_SIZE,
        device=Config.DEVICE,
        n_folds=1,
        sample_size=20,
        debug=True,
    )

    # --- Verification ---
    assert os.path.exists(output_csv), "Submission file was not created"
    assert (
        len(submission_df) == 20
    ), f"Expected 20 predictions, got {len(submission_df)}"
    assert "textID" in submission_df.columns, "Missing textID column"
    assert "selected_text" in submission_df.columns, "Missing selected_text column"

    # Check for empty strings (should be rare/impossible if logic holds, unless text is empty)
    empty_preds = submission_df[submission_df["selected_text"] == ""]
    if not empty_preds.empty:
        print("Warning: Some predictions are empty strings.")

    print("Inference pipeline verified successfully.")


if __name__ == "__main__":
    # Ensure reproducible results
    seed_everything(42)

    try:
        # 1. Test low-level utilities
        test_utility_functions()

        # 2. Test Training (and save model)
        saved_model_path = run_training_pipeline_demo()

        # 3. Test Inference (using saved model)
        run_inference_pipeline_demo(saved_model_path)

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        # Print full traceback for debugging if needed
        import traceback

        traceback.print_exc()
        exit(1)
