import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import get_loaders
from library.model import SentimentConditionedDeberta
from library.engine import train_fn, eval_fn


def run_demo():
    print("=== Starting Sentiment Extraction Pipeline Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment...")
    seed_everything(Config.SEED)

    # Override Config for a fast demo run
    Config.DEBUG = True  # Uses only 100 rows of data
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.ARTIFACT_DIR = "./working/demo_execution"
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")
    print(f"Artifact Dir: {Config.ARTIFACT_DIR}")

    # 2. Data Loading
    print("\n[2] Preparing DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # get_loaders handles caching and processing.
    # Since DEBUG=True, it will process a small subset and skip cache loading.
    train_loader, val_loader, test_loader = get_loaders(
        tokenizer, load_cached_data=False
    )

    # Validation: Check Loader
    assert len(train_loader) > 0, "Train loader should not be empty"
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # Inspect one batch
    sample_batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "offsets",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Batch missing key: {key}"

    print(f"Input IDs Shape: {sample_batch['input_ids'].shape}")
    # Shape should be [Batch_Size, Max_Len] -> [8, 128]
    assert sample_batch["input_ids"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = SentimentConditionedDeberta()
    model.to(Config.DEVICE)

    # Validation: Dummy Forward Pass
    print("Running dummy forward pass...")
    input_ids = sample_batch["input_ids"].to(Config.DEVICE)
    attention_mask = sample_batch["attention_mask"].to(Config.DEVICE)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask)

    # Check output shapes: [Batch, SeqLen]
    assert start_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Shape mismatch: {end_logits.shape}"
    print("Forward pass successful. Output shapes verified.")

    # 4. Training Loop
    print("\n[4] Starting Training Demo (1 Epoch)...")
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    avg_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
    print(f"Epoch 1 Training Loss: {avg_loss:.4f}")

    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss > 0, "Training loss should be positive"

    # 5. Evaluation & Inference Logic
    print("\n[5] Evaluating and Demonstrating Inference...")
    val_loss, val_start_logits, val_end_logits = eval_fn(
        val_loader, model, Config.DEVICE
    )
    print(f"Validation Loss: {val_loss:.4f}")

    # Load original validation data to compare predictions against text
    # Since DEBUG=True, get_loaders processed the first 100 rows of the validation file
    df_val = pd.read_csv(Config.VAL_FILE).head(100)

    # Let's decode the first prediction from the validation set
    idx = 0
    start_logits = val_start_logits[idx]
    end_logits = val_end_logits[idx]

    # Get offsets from the dataset (need to retrieve from loader again or iterate)
    # For this demo, we'll grab the batch corresponding to idx=0
    # Note: val_loader shuffle is False by default in the library for val/test?
    # Checking library/data.py: shuffle=False for val. Good.

    val_iter = iter(val_loader)
    first_batch = next(val_iter)
    offsets = first_batch["offsets"][idx].numpy()

    # Simple Argmax Decoding
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    # Heuristic: Ensure end >= start
    if end_idx < start_idx:
        end_idx = start_idx

    # Extract text using offsets
    # The offsets map token indices to character positions in the *original* (normalized) text
    pred_char_start = offsets[start_idx][0]
    pred_char_end = offsets[end_idx][1]

    original_text = str(df_val.iloc[idx]["text"])
    # Note: The model sees normalized text.
    # For strict submission, we usually extract from normalized and map back,
    # or just extract from original if offsets align well.
    # The library `process_data` normalizes text before tokenization.

    # For this demo, we slice the text roughly based on offsets.
    # In a real scenario, we would handle the normalization shift carefully.
    predicted_text = original_text[pred_char_start:pred_char_end]
    target_text = str(df_val.iloc[idx]["selected_text"])

    print(f"\n--- Sample Prediction (Index {idx}) ---")
    print(f"Original Text:   {original_text}")
    print(f"Sentiment:       {df_val.iloc[idx]['sentiment']}")
    print(f"Predicted Text:  '{predicted_text}'")
    print(f"Target Text:     '{target_text}'")

    score = jaccard(predicted_text, target_text)
    print(f"Jaccard Score:   {score:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
