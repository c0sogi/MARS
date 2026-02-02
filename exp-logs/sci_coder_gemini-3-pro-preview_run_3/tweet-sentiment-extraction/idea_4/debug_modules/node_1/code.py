import os
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, jaccard, get_selected_text
from library.dataset import get_data, TweetDataset
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup Environment
    # Override Config for speed
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.DEBUG = True  # This adds _debug to cache paths

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed_everything(Config.SEED)

    # 2. Data Preparation
    print("\n--- 1. Data Loading & Processing ---")

    # Load a tiny subset of the training metadata
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_META_PATH}")

    df_train_full = pd.read_csv(Config.TRAIN_META_PATH)
    # Take top 20 rows for demonstration speed
    df_demo = df_train_full.head(20).copy()
    print(f"Demo DataFrame shape: {df_demo.shape}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Use get_data to process/cache the dataset
    # We use a temporary cache path for this demo
    cache_path = os.path.join(Config.WORKING_DIR, "demo_cache")

    train_dataset = get_data(
        df_demo,
        tokenizer,
        max_len=Config.MAX_LEN,
        cache_path=cache_path,
        load_cached_data=False,  # Force processing
        is_train=True,
        filter_neutral=False,
        debug=False,
    )

    # Validation: Check Dataset Integrity
    print(f"Dataset length: {len(train_dataset)}")
    sample_item = train_dataset[0]

    # Check tensor shapes
    assert sample_item["input_ids"].shape == (
        Config.MAX_LEN,
    ), "input_ids shape mismatch"
    assert sample_item["attention_mask"].shape == (
        Config.MAX_LEN,
    ), "attention_mask shape mismatch"
    assert isinstance(
        sample_item["start_labels"].item(), int
    ), "start_label is not an int"
    assert isinstance(sample_item["end_labels"].item(), int), "end_label is not an int"

    print("Dataset verification passed: Tensor shapes are correct.")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for demo
        drop_last=True,
    )

    # 3. Model Initialization
    print("\n--- 2. Model Initialization ---")
    model = TweetModel()
    model.to(device)

    # Validation: Forward Pass
    dummy_batch = next(iter(train_loader))
    input_ids = dummy_batch["input_ids"].to(device)
    mask = dummy_batch["attention_mask"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask)

    print(f"Logits Shape: {start_logits.shape}")

    # Check output dimensions: (Batch_Size, Seq_Len)
    expected_shape = (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
    assert (
        start_logits.shape == expected_shape
    ), f"Expected start_logits shape {expected_shape}, got {start_logits.shape}"
    assert (
        end_logits.shape == expected_shape
    ), f"Expected end_logits shape {expected_shape}, got {end_logits.shape}"
    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- 3. Training Loop (1 Epoch) ---")

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    num_train_steps = int(len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Run training function
    avg_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"Training completed. Average Loss: {avg_loss:.4f}")

    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss > 0, "Training loss should be positive"

    # 5. Evaluation Loop Demonstration
    print("\n--- 4. Evaluation Loop ---")

    # Reuse train_loader as val_loader for simplicity in this demo
    val_loss, val_jaccard = eval_fn(train_loader, model, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Jaccard: {val_jaccard:.4f}")

    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score out of range [0, 1]"

    # 6. Utility Logic Verification
    print("\n--- 5. Utility Logic Verification ---")

    # Test Jaccard
    score_perfect = jaccard("hello world", "hello world")
    score_none = jaccard("hello world", "foo bar")
    score_partial = jaccard("hello world", "hello")

    assert score_perfect == 1.0, "Jaccard logic error (perfect match)"
    assert score_none == 0.0, "Jaccard logic error (no match)"
    assert 0.0 < score_partial < 1.0, "Jaccard logic error (partial match)"
    print("Jaccard function verified.")

    # Test get_selected_text (Post-processing)
    # Scenario: Neutral sentiment -> return full text
    text = "This is a neutral tweet."
    res_neutral = get_selected_text(text, [0], [0], "neutral", [])
    assert res_neutral == text, "Neutral sentiment should return full text"

    # Scenario: Positive sentiment -> use probabilities
    # We mock probabilities to select "good" from "It is good"
    # Text: "It is good"
    # Offsets (approx): "It"=(0,2), " is"=(3,5), " good"=(6,10)
    # Tokens: [CLS], It, is, good, [SEP]
    # Indices:  0,    1,  2,   3,    4

    mock_text = "It is good"
    mock_offsets = [(0, 0), (0, 2), (3, 5), (6, 10), (0, 0)]  # Simplified

    # Set high probs for index 3 (start) and 3 (end) -> "good"
    start_probs = np.array([0.0, 0.0, 0.0, 0.9, 0.1])
    end_probs = np.array([0.0, 0.0, 0.0, 0.9, 0.1])

    res_pos = get_selected_text(
        mock_text, start_probs, end_probs, "positive", mock_offsets
    )
    assert res_pos == "good", f"Expected 'good', got '{res_pos}'"

    print("Post-processing logic verified.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
