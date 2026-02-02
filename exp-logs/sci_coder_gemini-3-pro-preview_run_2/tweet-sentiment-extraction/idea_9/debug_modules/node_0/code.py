import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Import provided library components
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import get_data, create_loader
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_demo():
    print("=== Starting Tweet Sentiment Extraction Demo ===")

    # 1. Setup Configuration for Fast Execution
    print("\n[1] Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Verify Utility Logic
    print("\n[2] Verifying Jaccard Metric Logic...")
    s1 = "good morning"
    s2 = "good"
    score = jaccard(s1, s2)
    print(f"Jaccard('{s1}', '{s2}') = {score}")

    # Expected: Intersection is {'good'}, Union is {'good', 'morning'}. 1/2 = 0.5
    assert abs(score - 0.5) < 1e-6, "Jaccard calculation failed for partial overlap."
    assert jaccard("same", "same") == 1.0, "Jaccard failed for identical strings."
    assert jaccard("abc", "def") == 0.0, "Jaccard failed for disjoint strings."
    print("Jaccard logic verified.")

    # 3. Data Pipeline
    print("\n[3] Setting up Data Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load Training Data (Debug Mode)
    print(f"Loading training data from {Config.TRAINING_FILE}...")
    train_data = get_data(
        Config.TRAINING_FILE,
        tokenizer,
        Config.MAX_LEN,
        Config.CACHE_DIR,
        load_cached_data=False,  # Force processing for demo
    )

    # Load Validation Data (Debug Mode)
    print(f"Loading validation data from {Config.VALIDATION_FILE}...")
    val_data = get_data(
        Config.VALIDATION_FILE,
        tokenizer,
        Config.MAX_LEN,
        Config.CACHE_DIR,
        load_cached_data=False,
    )

    # Create Loaders
    train_loader = create_loader(
        train_data, tokenizer, Config.TRAIN_BATCH_SIZE, is_train=True
    )
    val_loader = create_loader(
        val_data, tokenizer, Config.VALID_BATCH_SIZE, is_train=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 4. Model Initialization & Shape Check
    print("\n[4] Initializing Model and Verifying Shapes...")
    model = TweetModel(Config.MODEL_PATH)
    model.to(device)

    # Fetch one batch to verify forward pass dimensions
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

    print(f"Input Shape: {input_ids.shape}")
    print(f"Start Logits Shape: {start_logits.shape}")
    print(f"End Logits Shape: {end_logits.shape}")

    # Assertions
    batch_size, seq_len = input_ids.shape
    assert start_logits.shape == (batch_size, seq_len), "Start logits shape mismatch"
    assert end_logits.shape == (batch_size, seq_len), "End logits shape mismatch"
    print("Model forward pass shape verified.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Optimizer setup
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=Config.LEARNING_RATE)
    num_train_steps = int(len(train_data) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # Train
    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"Epoch 1 Train Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss is NaN"

    # 6. Evaluation Demonstration
    print("\n[6] Running Evaluation...")
    val_loss, val_jaccard = eval_fn(val_loader, model, device)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Jaccard: {val_jaccard:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score out of bounds"

    # 7. Inference Demonstration (Manual)
    print("\n[7] Manual Inference Check on Single Example...")
    test_text = "The weather is absolutely fantastic today!"
    test_sentiment = "positive"

    # Tokenize
    encoded = tokenizer(
        test_sentiment,
        test_text,
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)
    offsets = encoded["offset_mapping"][0].cpu().numpy()

    # Predict
    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

    # Decode
    if end_idx < start_idx:
        end_idx = start_idx

    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]
    predicted_phrase = (
        test_text[start_char:end_char] if test_sentiment != "neutral" else test_text
    )

    print(f"Text: '{test_text}'")
    print(f"Sentiment: '{test_sentiment}'")
    print(f"Predicted Selected Text: '{predicted_phrase}'")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
